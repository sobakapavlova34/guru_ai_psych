from __future__ import annotations

import json
import logging
import re
from typing import Annotated, Any, Literal, Sequence, TypedDict

import operator
import traceback

logger = logging.getLogger(__name__)

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from knowledge_base import PROBLEM_KNOWLEDGE_BASE, format_problem_kb_for_prompt
from llm.bothub_client import bothub_client

BASE_SYSTEM_PROMPT = """Ты как друг — лёгкий, тёплый, без официальности. Общайся на «ты».
Можно просто выговориться, без давления. Человек может не знать своей проблемы — это ок.
Обязательно используй эмодзи, где это уместно!
ИНСТРУКЦИИ ДЛЯ СТИЛЯ РЕЧИ/TONE OF VOICE:
- НЕ общайся с пользователем как с пациентом
- Ты должен шутить! И иногда ты можешь даже грубо шутить ("ты знаешь, мне тоже иногда кажется, что все люди уроды"), НО ТОЛЬКО ЕСЛИ ПОЛЬЗОВАТЕЛЮ НУЖНА ТАКАЯ ПОДДЕРЖКА
- Используй эмодзи, если это уместно 
- Задавай детальные вопросы, которые выведут тебя на определение конкретной проблемы пользователя, которую нужно проработать
- НЕ БЛАГОДАРИ ЗА ОТВЕТЫ 

ПРИМЕР: 
User: я не знаю что мне делать меня все достало
PSYCHOLOGIST: Не всегда легко сформулировать, что на душе. Попробуйте начать с любого слова или чувства — я помогу развить мысль.
 
User: у меня проблема в карьере, я не знаю как мне найти себя
 PSYCHOLOGIST: Оо, да ты думаешь сейчас все сплошь и рядом устроенные и определившиеся? Каждый второй с этой проблемой, потому что мир отстойно устроен, но как есть, и давай его вместе победим!"""

OFF_TOPIC_KEYWORDS = (
    "проблем",
    "здоровь",
    "болит",
    "цел",
    "привело",
    "пришёл",
    "пришла",
    "беспоко",
    "тревож",
    "депресс",
    "стресс",
)


def _normalize_for_fixed_question_match(text: str) -> str:
    t = " ".join((text or "").lower().split())
    for a, b in (("слышала", "слышал"), ("слышали", "слышал")):
        t = t.replace(a, b)
    return t


def _fixed_question_already_in_content(fixed_question: str, content: str) -> bool:
    if not fixed_question or not content:
        return False
    if fixed_question in content:
        return True
    return _normalize_for_fixed_question_match(fixed_question) in _normalize_for_fixed_question_match(
        content
    )


class DialogState(TypedDict, total=False):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    conversation_topic: str
    stage: str
    turn_count: int
    tell_about_yourself_pending: int
    user_meta: dict[str, Any]
    scenario: str


def _get_effective_count(state: DialogState) -> int:
    messages = state.get("messages", [])
    user_msg_count = sum(1 for m in messages if isinstance(m, HumanMessage))
    pending = state.get("tell_about_yourself_pending", 0)
    return user_msg_count - pending


def _last_user_text(messages: Sequence[BaseMessage]) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return (m.content or "").strip()
    return ""


def _merge_user_meta_base(state: DialogState, effective_count: int) -> dict[str, Any]:
    """Сырой слепок: фрагмент реплики, черновое имя на шаге 1. Флаги — только через LLM."""
    meta = dict(state.get("user_meta") or {})
    text = _last_user_text(state.get("messages", []))
    if not text:
        return meta

    turns = list(meta.get("turns", []))
    turns.append({"intro_step": effective_count, "snippet": text[:500]})
    meta["turns"] = turns

    if effective_count == 1 and not meta.get("preferred_name"):
        meta["preferred_name"] = text.split()[0][:40] if text else ""

    return meta


def _llm_extract_intro_meta(
    llm: Any,
    *,
    intro_step: int,
    last_user_text: str,
    assistant_reply: str | None,
    accumulated_before: dict[str, Any],
) -> dict[str, Any]:
    """LLM интерпретирует последнюю реплику (шаги 1–5): только текстовые выводы, без флагов."""
    acc = json.dumps(accumulated_before, ensure_ascii=False)[:4000]
    ar = (assistant_reply or "").strip()
    assistant_block = (
        f"Ответ ассистента пользователю (уже сгенерирован):\n{ar}\n"
        if ar
        else "Ответ ассистента этому сообщению ещё не сформирован — опирайся только на реплику пользователя и мета.\n"
    )
    phase = (
        f"вводный диалог, шаг {intro_step} из 5"
        if intro_step <= 5
        else f"шаг {intro_step} — последняя реплика перед выбором сценария (после неё классификация)"
    )
    meta_prompt = f"""Ты аналитик реплики пользователя в чате с поддерживающим ботом ({phase}).

Уже накопленная мета до этой реплики (JSON):
{acc}

{assistant_block}
Последняя реплика пользователя:
«{last_user_text[:2000]}»

Флаги mentions_problem / problem_unclear / casual_chat здесь НЕ заполняй — они выставляются один раз после 5 вводных шагов по всей мете.

Верни СТРОГО один JSON (без текста вокруг):
{{
  "interpretation": "1–3 предложения: что пользователь выражает своими словами",
  "portrait_note": "одно короткое предложение: что добавить к накопленному портрету пользователя",
  "preferred_name": null или строка — только при шаге 1, если уверенно выделяется имя/обращение; иначе null
}}
"""
    raw = llm.invoke(
        [SystemMessage(content=meta_prompt), HumanMessage(content="Верни JSON.")]
    )
    raw_text = raw.content if hasattr(raw, "content") else str(raw)
    try:
        return _parse_json_object(raw_text)
    except (json.JSONDecodeError, ValueError):
        logger.warning("llm_extract_intro_meta: JSON не распарсен")
        return {
            "interpretation": "",
            "portrait_note": "",
            "preferred_name": None,
        }


def _finalize_intro_flags(llm: Any, user_meta: dict[str, Any]) -> tuple[dict[str, bool], str]:
    meta_for_prompt = {k: v for k, v in user_meta.items() if k not in ("flags", "classification")}
    blob = json.dumps(meta_for_prompt, ensure_ascii=False, indent=2)[:8000]
    prompt = f"""Ты аналитик вводной фазы диалога (ровно 5 первых пользовательских реплик по сценарию intro).

Ниже — накопленная метаинформация о пользователе: фрагменты реплик (turns), пошаговые интерпретации (step_insights), имя и т.д.
Оцени В ЦЕЛОМ по этим пяти шагам, каков запрос пользователя.

Верни СТРОГО один JSON:
{{
  "flags": {{
    "mentions_problem": true|false,
    "problem_unclear": true|false,
    "casual_chat": true|false
  }},
  "summary": "одно предложение: почему такие значения флагов"
}}

Смысл флагов (присваивается строго один флаг!):
- mentions_problem: по итогам 5 реплик видно, что человек опирается на конкретную проблему/ситуацию и хочет её прорабатывать.
- problem_unclear: проблема не сформулирована, но есть переживание, путаница, «не понимаю что не так».
- casual_chat: преобладает запрос на лёгкий контакт/поболтать без фокуса на разборе проблемы.

Мета:
{blob}
"""
    raw = llm.invoke([SystemMessage(content=prompt), HumanMessage(content="Верни JSON.")])
    raw_text = raw.content if hasattr(raw, "content") else str(raw)
    try:
        parsed = _parse_json_object(raw_text)
        flags = dict((parsed.get("flags") or {}))
        out = {
            "mentions_problem": bool(flags.get("mentions_problem")),
            "problem_unclear": bool(flags.get("problem_unclear")),
            "casual_chat": bool(flags.get("casual_chat")),
        }
        summary = str(parsed.get("summary", "")).strip()
        return out, summary
    except (json.JSONDecodeError, ValueError, TypeError):
        logger.warning("_finalize_intro_flags: JSON не распарсен, все false")
        return (
            {
                "mentions_problem": False,
                "problem_unclear": False,
                "casual_chat": False,
            },
            "",
        )


def _merge_llm_meta_into_accumulator(
    meta: dict[str, Any],
    intro_step: int,
    llm_payload: dict[str, Any],
) -> dict[str, Any]:
    out = dict(meta)

    insights = list(out.get("step_insights", []))
    insights.append(
        {
            "intro_step": intro_step,
            "interpretation": (llm_payload.get("interpretation") or "").strip(),
            "portrait_note": (llm_payload.get("portrait_note") or "").strip(),
        }
    )
    out["step_insights"] = insights

    pn = llm_payload.get("preferred_name")
    if intro_step == 1 and isinstance(pn, str) and pn.strip():
        out["preferred_name"] = pn.strip()[:80]

    return out


def _get_intro_instructions(
    effective_count: int,
) -> tuple[str, str | None, str]:
    if effective_count == 1:
        # q = "Ты что-нибудь слышал о том, как я работаю? Или просто хочешь поболтать?"
        return (
            f"""ЭТАП 1/5 — контакт. Только представься и согласуй обращение по имени и на ВЫ/на ТЫ.
- Пользователь написал имя или как к нему обращаться — отзеркаль стиль (Елена→Мария, Дэн→Кэт).
- Пользователь выбрал форму обращения на ВЫ/на ТЫ - сохрани эту форму обращения в диалоге»
ЗАПРЕЩЕНО: углубляться в проблемы, здоровье, семью, «что привело».""",
            None,
            "intro_1_contact",
        )
    if effective_count == 2:
        # fq = "Окей, давай познакомимся поближе? Напиши что-нибудь."
        return (
            f"""ЭТАП 2/5 — формат. Коротко объясни, как ты работаешь (поддержка, конфиденциальность, без оценок).»
ЗАПРЕЩЕНО: расспросы о проблеме и «что тебя беспокоит».""",
            None,
            "intro_2_format",
        )
    if effective_count == 3:
        return (
            """ЭТАП 3/5 — лёгкое присутствие. Коротко отреагируй на реплику.
Можно мягко спросить, как проходит день/настроение — одной фразой, без анкеты.
Не толкай к проблеме; если человек сам зашёл в тяжёлое — мягко признай и не углубляйся.""",
            None,
            "intro_3_mood",
        )
    if effective_count == 4:
        return (
            """ЭТАП 4/5 — контекст в общих чертах. Поблагодари за открытость.
Одним открытым вопросом можно спросить, что сейчас на уме в целом (без давления и без «какая проблема»).
Если человек не готов — прими это.""",
            None,
            "intro_4_context",
        )
    if effective_count == 5:
        return (
            """ЭТАП 5/5 — мост. Подведи к тому, что дальше вы вместе посмотрите, как лучше продолжить разговор
(в зависимости от того, что ему сейчас важно). Без обещаний терапии и без диагнозов.
Тон: тёплый, коротко. Следующий пользовательский ход — развилка сценариев (это сделает система).""",
            None,
            "intro_5_bridge",
        )
    return ("", None, "intro")


def _parse_json_object(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    if "```" in raw:
        parts = raw.split("```")
        for p in parts:
            p = p.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            if p.startswith("{"):
                raw = p
                break
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        raise ValueError("JSON не найден в ответе модели")
    return json.loads(m.group(0))


def create_dialog_graph(
    top_p: float = 1,
    max_completion_tokens: int = 2000,
    temperature: float | None = 0.2,
):
    llm = bothub_client(
        top_p=top_p,
        max_completion_tokens=max_completion_tokens,
        temperature=temperature,
    )

    def router_node(state: DialogState) -> DialogState:
        return {}

    def should_route(
        state: DialogState,
    ) -> Literal["greeting", "intro", "classify", "scenario_dialog", "ended"]:
        ec = _get_effective_count(state)
        if ec == 0:
            return "greeting"
        if ec <= 5:
            return "intro"
        if ec == 6:
            return "classify"
        if ec >= 80:
            return "ended"
        return "scenario_dialog"

    def greeting_node(state: DialogState) -> DialogState:
        turn_count = state.get("turn_count", 0) + 1
        greeting_text = """Привет! Я тут, чтобы поболтать и помочь разобраться, если что-то гложет.
Устраивайся поудобнее. Как тебя зовут?"""
        return {
            "messages": [AIMessage(content=greeting_text)],
            "stage": "greeting",
            "turn_count": turn_count,
            "user_meta": {},
            "scenario": "",
        }

    def intro_node(state: DialogState) -> DialogState:
        messages = state["messages"]
        turn_count = state.get("turn_count", 0) + 1
        pending = state.get("tell_about_yourself_pending", 0)
        raw_ec = _get_effective_count(state)
        effective_count = raw_ec

        index = effective_count - 3
        is_off_topic = False
        if 3 <= effective_count <= 5 and index > 0:
            lt = _last_user_text(messages).lower()
            if any(kw in lt for kw in OFF_TOPIC_KEYWORDS):
                is_off_topic = True
                effective_count -= 1

        prev_meta = dict(state.get("user_meta") or {})
        base_meta = _merge_user_meta_base(state, min(raw_ec, 5))

        stage_instructions, fixed_question, stage_name = _get_intro_instructions(effective_count)
        system_prompt = f"{BASE_SYSTEM_PROMPT}\n\n{stage_instructions}"
        system_message = SystemMessage(content=system_prompt)
        messages_with_system = [system_message] + list(messages)

        llm_response = llm.invoke(messages_with_system)
        content = llm_response.content if hasattr(llm_response, "content") else str(llm_response)
        if fixed_question and not _fixed_question_already_in_content(fixed_question, content):
            content = f"{content.strip()}\n\n{fixed_question}" if content.strip() else fixed_question

        meta_payload = _llm_extract_intro_meta(
            llm,
            intro_step=min(raw_ec, 5),
            last_user_text=_last_user_text(messages),
            assistant_reply=content,
            accumulated_before=prev_meta,
        )
        user_meta = _merge_llm_meta_into_accumulator(base_meta, min(raw_ec, 5), meta_payload)

        if raw_ec == 5:
            flags_dict, flags_summary = _finalize_intro_flags(llm, user_meta)
            user_meta["flags"] = flags_dict
            if flags_summary:
                user_meta["flags_rationale"] = flags_summary
            user_meta["intro_flags_finalized"] = True

        result: dict[str, Any] = {
            "messages": [AIMessage(content=content)],
            "stage": stage_name,
            "turn_count": turn_count,
            "user_meta": user_meta,
            "tell_about_yourself_pending": pending + 1 if is_off_topic else 0,
        }
        return result

    def classify_node(state: DialogState) -> DialogState:
        messages = state["messages"]
        turn_count = state.get("turn_count", 0) + 1
        user_meta = _merge_user_meta_base(state, 6)

        kb_outline = format_problem_kb_for_prompt()
        meta_json = json.dumps(user_meta, ensure_ascii=False, indent=2)[:6000]

        classify_system = f"""Ты служебный классификатор сценария продолжения диалога.
Доступные типы проблем из базы (для ориентира, не для диагноза):
{kb_outline}

Метаинформация, собранная за первые 5 реплик пользователя:
{meta_json}

Верни ОДИН JSON-объект (без текста вокруг) вида:
{{
  "scenario": "1" | "2" | "3",
  "reply": "<короткий тёплый ответ пользователю по-русски: подтверди выбранный фокус и мягко начни следующий этап>",
  "note": "<одно предложение: почему такой сценарий>"
}}

Критерии:
- "1" — пользователь в явном виде называет проблему/ситуацию и готов её прорабатывать.
- "2" — проблема не сформулирована, но есть сильные переживания/путаница/«не понимаю, что не так».
- "3" — в основном хочет просто поговорить, без запроса на разбор проблемы (всё равно держи в голове возможные темы из базы на будущее)."""

        classify_human = "Проанализируй последние реплики и метаинформацию. Выбери сценарий и сформируй reply."
        cls_messages = [
            SystemMessage(content=classify_system),
            HumanMessage(content=classify_human),
        ]
        raw = llm.invoke(cls_messages)
        raw_text = raw.content if hasattr(raw, "content") else str(raw)
        try:
            parsed = _parse_json_object(raw_text)
        except (json.JSONDecodeError, ValueError):
            parsed = {
                "scenario": "3",
                "reply": "Понял тебя. Давай спокойно продолжим — пиши, как тебе комфортно.",
                "note": "fallback: JSON не распарсен",
            }

        scenario = str(parsed.get("scenario", "3")).strip()
        if scenario not in ("1", "2", "3"):
            scenario = "3"
        reply = str(parsed.get("reply", "Давай продолжим в том темпе, который тебе ок.")).strip()
        note = str(parsed.get("note", "")).strip()

        user_meta["classification"] = {"scenario": scenario, "note": note, "raw": raw_text[:2000]}

        return {
            "messages": [AIMessage(content=reply)],
            "stage": f"classified_scenario_{scenario}",
            "turn_count": turn_count,
            "user_meta": user_meta,
            "scenario": scenario,
            "tell_about_yourself_pending": 0,
        }

    def scenario_node(state: DialogState) -> DialogState:
        messages = state["messages"]
        turn_count = state.get("turn_count", 0) + 1
        scenario = (state.get("scenario") or "3").strip()
        if scenario not in ("1", "2", "3"):
            scenario = "3"

        kb_text = format_problem_kb_for_prompt()
        keys = ", ".join(PROBLEM_KNOWLEDGE_BASE.keys()) if PROBLEM_KNOWLEDGE_BASE else "(пусто)"

        if scenario == "1":
            scenario_block = (
                "СЦЕНАРИЙ 1 — пользователь называет проблему и хочет её проработать.\n"
                "Опирайся на типы проблем из базы ниже: аккуратно связывай реплики пользователя с подходящими типами, "
                "предлагай уточняющие вопросы из области litmus/описания, не навязывай ярлык.\n"
                "Не ставь диагнозов; при риске для жизни — скажи про экстренную помощь."
            )
        elif scenario == "2":
            scenario_block = (
                "СЦЕНАРИЙ 2 — проблема пока не ясна, но есть переживание.\n"
                "Помоги человеку чуть яснее назвать, что происходит (чувства, ситуации, ожидания). "
                "Используй типы из базы как ориентиры «возможных направлений», не как вердикт.\n"
                "Задавай короткие открытые вопросы, один за раз."
            )
        else:
            scenario_block = (
                "СЦЕНАРИЙ 3 — в основном свободный разговор.\n"
                "Поддерживай контакт; если всплывают устойчивые темы, мягко можно связать их с типами из базы "
                "как гипотезы для размышления, без давления."
            )

        system_prompt = f"""{BASE_SYSTEM_PROMPT}

ТЕКУЩИЙ РЕЖИМ (после 6-го сообщения пользователя зафиксирован сценарий {scenario}).
{scenario_block}

СПРАВОЧНИК ТИПОВ ПРОБЛЕМ (PROBLEM_KNOWLEDGE_BASE, ключи: {keys}):
{kb_text}

Мета (кратко, для контекста): {json.dumps(state.get("user_meta", {}), ensure_ascii=False)[:2500]}
"""

        system_message = SystemMessage(content=system_prompt)
        messages_with_system = [system_message] + list(messages)
        llm_response = llm.invoke(messages_with_system)
        content = llm_response.content if hasattr(llm_response, "content") else str(llm_response)

        return {
            "messages": [AIMessage(content=content)],
            "stage": f"scenario_{scenario}_work",
            "turn_count": turn_count,
        }

    def ended_node(state: DialogState) -> DialogState:
        turn_count = state.get("turn_count", 0) + 1
        closing = (
            "На сегодня достаточно. Спасибо, что поделился. "
            "Если захочешь продолжить — напиши снова."
        )
        return {
            "messages": [AIMessage(content=closing)],
            "stage": "ended",
            "turn_count": turn_count,
        }

    graph = StateGraph(DialogState)
    graph.add_node("router", router_node)
    graph.add_node("greeting", greeting_node)
    graph.add_node("intro", intro_node)
    graph.add_node("classify", classify_node)
    graph.add_node("scenario_dialog", scenario_node)
    graph.add_node("ended", ended_node)

    graph.set_entry_point("router")
    graph.add_conditional_edges(
        "router",
        should_route,
        {
            "greeting": "greeting",
            "intro": "intro",
            "classify": "classify",
            "scenario_dialog": "scenario_dialog",
            "ended": "ended",
        },
    )
    graph.add_edge("greeting", END)
    graph.add_edge("intro", END)
    graph.add_edge("classify", END)
    graph.add_edge("scenario_dialog", END)
    graph.add_edge("ended", END)

    return graph.compile()


def run_dialog(
    top_p: float = 1,
    max_completion_tokens: int = 2000,
    temperature: float | None = 0.2,
):
    if not logging.root.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(levelname)s: %(message)s",
        )
    dialog_graph = create_dialog_graph(
        top_p=top_p,
        max_completion_tokens=max_completion_tokens,
        temperature=temperature,
    )
    MAX_TURNS = 50

    state: DialogState = {
        "messages": [],
        "conversation_topic": "",
        "stage": "chat",
        "turn_count": 0,
        "user_meta": {},
        "scenario": "",
    }

    first_run = True

    while True:
        if first_run:
            try:
                result = dialog_graph.invoke(state)
                last_message = result["messages"][-1]
                if isinstance(last_message, AIMessage):
                    print(f"Бот: {last_message.content}\n")
                state = {
                    "messages": result["messages"],
                    "conversation_topic": state.get("conversation_topic", ""),
                    "stage": result.get("stage", "chat"),
                    "turn_count": result.get("turn_count", 0),
                    "tell_about_yourself_pending": result.get("tell_about_yourself_pending", 0),
                    "user_meta": result.get("user_meta", {}),
                    "scenario": result.get("scenario", ""),
                }
                first_run = False
            except Exception:
                traceback.print_exc()
                break

        user_input = input("Вы: ").strip()
        if not user_input:
            continue

        state["messages"].append(HumanMessage(content=user_input))

        if _get_effective_count(state) == 6:
            logger.info(
                "Шаг 6: переход к классификации — готов определить фокус диалога и тип запроса "
                "(явная проблема / неясное переживание / в основном разговор)."
            )

        try:
            result = dialog_graph.invoke(state)
            last_message = result["messages"][-1]
            current_turn = result.get("turn_count", 0)
            sc = result.get("scenario", "")

            if str(result.get("stage", "")).startswith("classified_scenario_"):
                note = (result.get("user_meta") or {}).get("classification", {}).get("note", "")
                logger.info(
                    "Классификация после шага 6 завершена: сценарий=%s.%s",
                    sc or "?",
                    f" Пояснение: {note}" if note else "",
                )

            if isinstance(last_message, AIMessage):
                print(f"Бот: {last_message.content}")
                extra = f" [сценарий: {sc}]" if sc else ""
                print(f"[Шаг {current_turn}/{MAX_TURNS}]{extra}\n")

            if current_turn >= MAX_TURNS:
                print("До свидания!")
                break

            state = {
                "messages": result["messages"],
                "conversation_topic": state.get("conversation_topic", ""),
                "stage": result.get("stage", state.get("stage", "chat")),
                "turn_count": result.get("turn_count", 0),
                "tell_about_yourself_pending": result.get("tell_about_yourself_pending", 0),
                "user_meta": result.get("user_meta", {}),
                "scenario": result.get("scenario", ""),
            }
        except Exception:
            traceback.print_exc()
            break


if __name__ == "__main__":
    run_dialog(top_p=1, max_completion_tokens=2000, temperature=0.2)
