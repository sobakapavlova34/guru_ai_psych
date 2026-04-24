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

from knowledge_base import (
    PROBLEM_KNOWLEDGE_BASE,
    aggregate_kb_trigger_scores,
    collect_kb_trigger_hits,
    format_problem_kb_for_prompt,
    get_litmus_question_for_problem,
    get_problem_description,
    top_problem_keys_by_score,
)
from llm.bothub_client import bothub_client

BASE_SYSTEM_PROMPT = """Ты — автопсихолог, друг, с которым можно поговорить просто и легко, без лишней официальности. Ты умеешь слушать и помогать, не надавливая. Пользователь может говорить что угодно — даже если не понимает свою проблему, это нормально.

ОБЯЗАТЕЛЬНОЕ:

- Ты всегда обращаешься так, как выбрал пользователь — на «ты» или на «вы». Если выбрал «вы», то обращаешься строго на «вы» до конца диалога.
- Эмодзи должны использоваться, когда это уместно, чтобы поддержать дружеский тон
- Шутки — обязательно! Юмор лёгкий, иногда можно подшутить даже немного жёстко, но только если пользователь явно этого хочет/нуждается в такой поддержке
- Никогда не хвалишь пользователя просто так, не спасибо, не «молодец» — ты не учитель и не терапевт, ты друг, который слушает или вместе с ним ищет выход
- Всегда задаёшь наводящие, подробные вопросы, чтобы помочь пользователю понять и выразить свою проблему точнее
- Не забывай про лёгкость, тепло и отсутствие давления — тебя можно просто использовать как «подушку» для слов, или для конкретной помощи

Пример стиля диалога:

PSYCHOLOGIST
Привет! Я — автопсихолог 🤓 Можем просто поговорить или попробовать разобраться в чём-то. На «ты» или на «вы» предпочтёшь?

User
на вы

PSYCHOLOGIST
Тогда как к вам обращаться? 😊

User
Марина

PSYCHOLOGIST
Что привело вас сюда? Хотите выговориться или ищете конкретный совет?

User
Просто устала и ничего не понимаю.

PSYCHOLOGIST
Усталость — штука коварная, иногда кажется, что мир специально решил вас проверить на выносливость. Что в последнее время чаще всего вызывает эти ощущения?

Так должен выглядеть стиль и правила твоего общения с пользователем.

"""

STYLE_CONSTRAINTS_BLOCK = """ДОПОЛНИТЕЛЬНО ПО СТИЛЮ (выше приоритет обычных вежливых формулировок):
- Не используй «отлично», «супер», «замечательно» и подобные оценочные междометия.
- Не повторяй одну и ту же мысль дважды разными словами в одном сообщении; не дублируй название темы и её перефраз как «другую формулировку той же проблемы».
- Если говоришь о гипотезе или теме — один раз и по делу, без лишних синонимов подряд.
"""


def _format_hypothesis_tied_list(tied_keys: list[str]) -> str:
    lines: list[str] = []
    for i, k in enumerate(tied_keys, 1):
        desc = get_problem_description(k)
        lines.append(f"{i}. «{k}» — {desc}")
    return "\n".join(lines)


def _kb_hypothesis_keys_from_meta(um: dict[str, Any]) -> list[str]:
    """Ключи из базы: при ничьей — все tied; иначе один hypothesized_problem_key."""
    keys = list(um.get("hypothesis_tied_keys") or [])
    if not keys:
        hk = (um.get("hypothesized_problem_key") or "").strip()
        if hk:
            keys = [hk]
    seen: set[str] = set()
    out: list[str] = []
    for k in keys:
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out


def _collect_hypothesis_catalog(um: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Все гипотезы/направления для показа пользователю: ориентиры из базы (по мете)
    + короткие гипотезы аналитика по шагам (psychologist_hypotheses).
    """
    items: list[dict[str, Any]] = []
    seen_norm: set[str] = set()

    def _norm(s: str) -> str:
        return " ".join(s.lower().split())[:240]

    def add_item(text: str, *, source: str, key: str | None = None, hid: str | None = None) -> None:
        text = (text or "").strip()
        if not text:
            return
        n = _norm(text)
        if n in seen_norm:
            return
        seen_norm.add(n)
        iid = hid or f"{source}:{key or n[:32]}"
        row: dict[str, Any] = {"id": iid, "text": text, "source": source}
        if key:
            row["key"] = key
        items.append(row)

    for k in _kb_hypothesis_keys_from_meta(um):
        if k in PROBLEM_KNOWLEDGE_BASE:
            add_item(
                f"«{k}» — {get_problem_description(k)}",
                source="kb",
                key=k,
                hid=f"kb:{k}",
            )

    for h in um.get("psychologist_hypotheses") or []:
        if isinstance(h, dict):
            t = (h.get("text") or "").strip()
            if t:
                step = h.get("intro_step", "")
                add_item(t, source="llm", hid=f"llm:{step}:{_norm(t)[:24]}")
        elif isinstance(h, str) and h.strip():
            add_item(h.strip(), source="llm", hid=f"llm:{_norm(h)[:24]}")

    return items


def _format_catalog_numbered_for_prompt(catalog: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for i, it in enumerate(catalog, 1):
        lines.append(f"{i}. {it.get('text', '')}")
    return "\n".join(lines)


def _hypothesis_listing_block(catalog_numbered_text: str) -> str:
    return f"""БЛОК ВЫБОРА ФОКУСА (первый ответ в основной фазе; в мете hypothesis_choice_offered ещё false):

Начни с отдельной строки-заголовка:
── Гипотезы ──

Ниже — **все** направления, собранные по диалогу в метаданных (база знаний + гипотезы аналитика по шагам). Покажи пользователю **полный** пронумерованный список; формулировки гипотез нужно смягчить и переформулировать (чтобы в сообщении не было "пользователь считает/думает..."), но не схлопывай несколько пунктов в один и не убирай направления.

{catalog_numbered_text}

Попроси выбрать **номер** или описать своими словами, с каким направлением сейчас важнее поработать.

ЗАПРЕЩЕНО в этом же сообщении: техники самопомощи, «режимы проработки», разбор по шагам — это только **после** выбора фокуса (следующий ход)."""


def _work_modes_system_block(scenario: str) -> str:
    """Режимы проработки: первый практический блок после гипотезы."""
    if scenario == "3":
        return """БЛОК РАБОТЫ (первый ответ в основной фазе диалога; в мете ещё не предлагали режимы — см. work_mode_offered):

Начни с отдельной строки-заголовка (чтобы было видно, где начинается работа с запросом):
── Проработка ──

Кратко без повторов и без «отлично». Предложи **на выбор** уровень, на котором человеку сейчас комфортно:
1) просто живой разговор и поддержка без программы;
2) несколько коротких техник самопомощи или упражнений в переписке;
3) более структурный разговор (по шагам уточняем, что происходит), если захочет углубиться.

Спроси, что ближе сейчас; другой режим можно выбрать позже."""

    return """БЛОК РАБОТЫ (первый ответ в основной фазе диалога; в мете ещё не предлагали режимы — см. work_mode_offered):

Начни с отдельной строки-заголовка (чтобы было видно, где начинается работа с темой):
── Проработка ──

Кратко отрази выбор фокуса из прошлого сообщения пользователя (см. мету); не пересказывай весь список гипотез заново.
Если в мете hypothesis_choice_offered=true — человек уже выбирал из блока «── Гипотезы ──»; переходи к режимам работы.

Предложи **на выбор** три режима проработки (номера 1–3):
1) **Техники самопомощи** — короткие упражнения и приёмы прямо в чате;
2) **Консультативный разговор** — вместе думаем, без жёсткого сценария;
3) **Разбор по шагам** — последовательно уточняем ситуацию, мысли, чувства (глубже, чем просто болтовня).

Спроси, какой режим ближе **сейчас**; при смене настроения режим можно сменить позже."""


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

    # совпадения с trigger_words из базы знаний (подстроки в реплике пользователя)
    if effective_count >= 3:
        kb_hits = collect_kb_trigger_hits(text)
        if kb_hits:
            hist = list(meta.get("kb_trigger_hits", []))
            hist.append({"intro_step": effective_count, "matches": kb_hits})
            meta["kb_trigger_hits"] = hist

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
  "preferred_name": null или строка — только при шаге 1, если уверенно выделяется имя/обращение; иначе null,
  "hypotheses": ["0–2 коротких строк: твои гипотезы о возможных темах/направлениях по ЭТОЙ реплике, если уместно; иначе []"]
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
            "hypotheses": [],
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

    raw_hyp = llm_payload.get("hypotheses")
    if isinstance(raw_hyp, list):
        existing = list(out.get("psychologist_hypotheses") or [])
        seen_lower = {
            (x.get("text") or "").strip().lower()
            for x in existing
            if isinstance(x, dict)
        }
        for item in raw_hyp:
            if not isinstance(item, str):
                continue
            t = item.strip()[:400]
            if not t or t.lower() in seen_lower:
                continue
            seen_lower.add(t.lower())
            existing.append({"intro_step": intro_step, "text": t})
        if existing:
            out["psychologist_hypotheses"] = existing

    return out


def _get_intro_instructions(
    effective_count: int,
) -> tuple[str, str | None, str]:
    if effective_count == 1:
        # q = "Ты что-нибудь слышал о том, как я работаю? Или просто хочешь поболтать?"
        return (
            f"""ЭТАП 1/5 — установление контакта, имя пользователя. 
- Пользователь выбрал форму обращения на ВЫ/на ТЫ - сохрани эту форму обращения в диалоге!
- ОБЯЗАТЕЛЬНО СПРОСИ ИМЯ ПОЛЬЗОВАТЕЛЯ, ИСПОЛЬЗУЯ ВЫБРАННУЮ ФОРМУ ОБРАЩЕНИЯ (на ВЫ/на ТЫ)
ЗАПРЕЩЕНО: углубляться в проблемы, здоровье, семью, «что привело».""", 
            None,
            "intro_1_contact",
        )
    if effective_count == 2:
        # fq = "Окей, давай познакомимся поближе? Напиши что-нибудь."
        return (
            f"""ЭТАП 2/5 — установление цели обращения пользователя к автопсихологу. 
            Спроси пользователя его цель обращения: чтобы просто выговориться или разобраться в какой-то ситуации/проблеме?
            - ОБЯЗАТЕЛЬНО ИСПОЛЬЗУЙ ВЫБРАННУЮ ФОРМУ ОБРАЩЕНИЯ (на ВЫ/на ТЫ)
            - Коротко объясни, как ты работаешь (поддержка, конфиденциальность, без оценок).
            ЗАПРЕЩЕНО: расспросы о проблеме и «что тебя беспокоит».""", 
            None,
            "intro_2_format",
        )
    if effective_count == 3:
        return (
            """ЭТАП 3/5 — начало "исследовательских вопросов" на предмет поиска определения проблемы пользователя.
            - Задай уточнящий исследовательский вопрос по проблеме пользователя
            - ОБЯЗАТЕЛЬНО ИСПОЛЬЗУЙ ВЫБРАННУЮ ФОРМУ ОБРАЩЕНИЯ (на ВЫ/на ТЫ)
            ЗАПРЕЩЕНО: "толкать" к проблеме - если человек сам зашёл в тяжёлое, мягко признай и не углубляйся.""",
            None,
            "intro_3_mood",
        )
    if effective_count == 4:
        return (
            """ЭТАП 4/5 — контекст в общих чертах. Признай, что человек поделился — без «спасибо» и без похвалы «молодец».
Одним открытым вопросом можно спросить, что сейчас на уме в целом (без давления и без «какая проблема»).
Если человек не готов — прими это.
- ОБЯЗАТЕЛЬНО ИСПОЛЬЗУЙ ВЫБРАННУЮ ФОРМУ ОБРАЩЕНИЯ (на ВЫ/на ТЫ)""",
            None,
            "intro_4_context",
        )
    if effective_count == 5:
        return (
            """ЭТАП 5/5 — мост. Подведи к тому, что дальше вы вместе посмотрите, как лучше продолжить разговор
(в зависимости от того, что ему сейчас важно). Без обещаний терапии и без диагнозов.
Следующий пользовательский ход — развилка сценариев (это сделает система).
- ОБЯЗАТЕЛЬНО ИСПОЛЬЗУЙ ВЫБРАННУЮ ФОРМУ ОБРАЩЕНИЯ (на ВЫ/на ТЫ)""",
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


def _llm_tie_break_problem(
    llm: Any, candidates: list[str], meta_blob: str, scores: dict[str, int]
) -> str:
    opts = "\n".join(
        f"- `{k}`: {get_problem_description(k)} (срабатываний триггеров: {scores.get(k, 0)})"
        for k in candidates
    )
    prompt = f"""По мета пользователя и числу срабатываний trigger_words выбери ОДИН наиболее подходящий тип проблемы.

Кандидаты (одинаковый счёт по триггерам):
{opts}

Мета (JSON):
{meta_blob}

Верни СТРОГО JSON: {{"problem_key": "<один из ключей кандидатов дословно>"}}"""
    raw = llm.invoke([SystemMessage(content=prompt), HumanMessage(content="Верни JSON.")])
    raw_text = raw.content if hasattr(raw, "content") else str(raw)
    try:
        parsed = _parse_json_object(raw_text)
        key = str(parsed.get("problem_key", "")).strip()
        if key in candidates:
            return key
    except (json.JSONDecodeError, ValueError, TypeError):
        logger.warning("_llm_tie_break_problem: JSON не распарсен")
    return candidates[0]


def _llm_pick_problem_no_hits(llm: Any, meta_blob: str) -> str:
    catalog = "\n".join(
        f"- `{k}`: {get_problem_description(k)}" for k in PROBLEM_KNOWLEDGE_BASE
    )
    keys = list(PROBLEM_KNOWLEDGE_BASE.keys())
    prompt = f"""По мета пользователя после вводной фазы (5 реплик) выбери ОДИН тип проблемы из справочника, который больше всего похож на запрос пользователя.

Типы (ключи должны совпадать дословно):
{catalog}

Мета:
{meta_blob}

Верни СТРОГО JSON: {{"problem_key": "<один из ключей списка>"}}"""
    raw = llm.invoke([SystemMessage(content=prompt), HumanMessage(content="Верни JSON.")])
    raw_text = raw.content if hasattr(raw, "content") else str(raw)
    try:
        parsed = _parse_json_object(raw_text)
        key = str(parsed.get("problem_key", "")).strip()
        if key in keys:
            return key
    except (json.JSONDecodeError, ValueError, TypeError):
        logger.warning("_llm_pick_problem_no_hits: JSON не распарсен")
    return keys[0] if keys else ""


def _llm_generate_litmus_question(llm: Any, problem_key: str, description: str) -> str:
    prompt = f"""Сформулируй один короткий уточняющий вопрос пользователю по-русски, чтобы проверить гипотезу о проблеме.

Тип (ключ): {problem_key}
Описание типа: {description}

Верни СТРОГО JSON: {{"question": "<одно предложение-вопрос>"}}"""
    raw = llm.invoke([SystemMessage(content=prompt), HumanMessage(content="Верни JSON.")])
    raw_text = raw.content if hasattr(raw, "content") else str(raw)
    try:
        parsed = _parse_json_object(raw_text)
        q = str(parsed.get("question", "")).strip()
        if q:
            return q
    except (json.JSONDecodeError, ValueError, TypeError):
        logger.warning("_llm_generate_litmus_question: JSON не распарсен")
    return f"Насколько тебе откликается то, что мы говорим про: {description[:200]}?"


def _resolve_hypothesis_problem(llm: Any, user_meta: dict[str, Any]) -> dict[str, Any]:
    scores = aggregate_kb_trigger_scores(user_meta)
    meta_blob = json.dumps(
        {k: v for k, v in user_meta.items() if k not in ("flags", "classification")},
        ensure_ascii=False,
    )[:6000]

    candidates = top_problem_keys_by_score(scores)
    tied_hypothesis_keys = list(candidates)
    hypothesis_multi = len(candidates) > 1
    reason = ""

    if len(candidates) == 1:
        chosen = candidates[0]
        reason = f"максимум совпадений trigger_words: {scores.get(chosen, 0)}"
    elif len(candidates) > 1:
        chosen = _llm_tie_break_problem(llm, candidates, meta_blob, scores)
        reason = f"ничья по счёту {candidates}, черновик для меты — LLM: {chosen}"
    else:
        chosen = _llm_pick_problem_no_hits(llm, meta_blob)
        tied_hypothesis_keys = [chosen] if chosen else []
        hypothesis_multi = False
        reason = "нет совпадений trigger_words, выбор LLM по мета"

    if chosen not in PROBLEM_KNOWLEDGE_BASE:
        chosen = next(iter(PROBLEM_KNOWLEDGE_BASE), "")

    litmus = ""
    if not hypothesis_multi and chosen:
        litmus = get_litmus_question_for_problem(chosen)
        if not litmus:
            litmus = _llm_generate_litmus_question(
                llm, chosen, get_problem_description(chosen)
            )

    return {
        "problem_key": chosen,
        "litmus_question": litmus.strip(),
        "reason": reason,
        "scores": scores,
        "hypothesis_multi": hypothesis_multi,
        "tied_hypothesis_keys": tied_hypothesis_keys,
    }


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
        greeting_text = """Привет! 
        Я - автопсихолог🤓 Я могу помочь разобраться с проблемой, а могу просто выслушать и поддержать. Ведь иногда просто хочется быть услышанным без всякого глубокого анализа🫶
        Будем на "ты" или на "вы"? 
        """
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

        index = effective_count - 3 # делаем счет с нуля для "подблока" сценария, не считая первые два шага - знакомство и "как я работаю"
        is_off_topic = False
        if 3 <= effective_count <= 5 and index > 0: # не первый из этих трёх подшагов, т.е. effective_count не равен 3 (потому что при 3: index = 0)
            lt = _last_user_text(messages).lower()
            if any(kw in lt for kw in OFF_TOPIC_KEYWORDS):
                is_off_topic = True
                effective_count -= 1

        prev_meta = dict(state.get("user_meta") or {})
        base_meta = _merge_user_meta_base(state, min(raw_ec, 5))

        hyp_key = ""
        litmus_override: str | None = None
        hypothesis_multi = False
        tied_keys: list[str] = []
        if raw_ec == 5:
            hypo = _resolve_hypothesis_problem(llm, base_meta)
            hyp_key = str(hypo.get("problem_key") or "").strip()
            litmus_override = (hypo.get("litmus_question") or "").strip() or None
            hypothesis_multi = bool(hypo.get("hypothesis_multi"))
            tied_keys = [str(x) for x in (hypo.get("tied_hypothesis_keys") or []) if x]
            base_meta["hypothesized_problem_key"] = hyp_key
            base_meta["hypothesis_litmus_question"] = litmus_override or ""
            base_meta["hypothesis_reason"] = str(hypo.get("reason", ""))
            base_meta["hypothesis_scores"] = hypo.get("scores") or {}
            base_meta["hypothesis_multi"] = hypothesis_multi
            base_meta["hypothesis_tied_keys"] = tied_keys

        stage_instructions, fixed_question, stage_name = _get_intro_instructions(effective_count)
        if raw_ec == 5 and hyp_key:
            if hypothesis_multi and tied_keys:
                stage_instructions += (
                    "\n\nЭТАП 5/5 — по смыслу и триггерам возможно **несколько** направлений (это не диагноз).\n"
                    "Ответь коротко, без «отлично» и без длинного перечисления. "
                    "**Полный список всех гипотез** (из базы + накопленных по шагам) и **выбор фокуса по номеру** "
                    "будут в первом сообщении основной фазы после классификации — не дублируй здесь пронумерованный каталог.\n"
                    "Можно в конце одним предложением сказать, что на следующем шаге вместе выберете фокус из полного списка.\n"
                )
                fixed_question = None
            elif litmus_override:
                stage_instructions += (
                    f"\n\nЭТАП 5/5 — ориентир по смыслу реплик (гипотеза): «{hyp_key}».\n"
                    "Стиль: без оценочных похвал, без синонимичных повторов.\n"
                    "В конце ответа ОБЯЗАТЕЛЬНО задай точно этот уточняющий вопрос:\n"
                    f"«{litmus_override}»\n"
                    "Добавь отдельное короткое предложение-вопрос: хочет ли человек **конкретно проработать** эту тему "
                    "далее или пока достаточно такого уровня разговора?"
                )
                fixed_question = litmus_override

        system_prompt = f"{BASE_SYSTEM_PROMPT}\n\n{STYLE_CONSTRAINTS_BLOCK}\n\n{stage_instructions}"
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
{STYLE_CONSTRAINTS_BLOCK}

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

        uec = _get_effective_count(state)
        um = dict(state.get("user_meta") or {})

        kb_text = format_problem_kb_for_prompt()
        keys = ", ".join(PROBLEM_KNOWLEDGE_BASE.keys()) if PROBLEM_KNOWLEDGE_BASE else "(пусто)"

        if scenario == "1":
            scenario_block = (
                "СЦЕНАРИЙ 1 — пользователь называет проблему и хочет её проработать.\n"
                "Система сначала может вывести блок «── Гипотезы ──» (полный список из меты), затем — «── Проработка ──» с режимами; следуй инструкциям в конце системного сообщения.\n"
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

        catalog = _collect_hypothesis_catalog(um)
        scenario_turn = uec - 6  # 1 = первый ответ бота после classify

        hypothesis_block = ""
        work_offer = ""
        meta_updates: dict[str, Any] = {}

        if scenario_turn == 1:
            if catalog and not um.get("hypothesis_choice_offered"):
                hypothesis_block = "\n\n" + _hypothesis_listing_block(
                    _format_catalog_numbered_for_prompt(catalog)
                )
                meta_updates["hypothesis_choice_offered"] = True
            elif not catalog and not um.get("work_mode_offered"):
                work_offer = "\n\n" + _work_modes_system_block(scenario)
                meta_updates["work_mode_offered"] = True
                meta_updates["hypothesis_choice_resolved"] = True
        elif not um.get("work_mode_offered"):
            work_offer = "\n\n" + _work_modes_system_block(scenario)
            meta_updates["work_mode_offered"] = True
            meta_updates["hypothesis_choice_resolved"] = True

        system_prompt = f"""{BASE_SYSTEM_PROMPT}

{STYLE_CONSTRAINTS_BLOCK}

ТЕКУЩИЙ РЕЖИМ (после 6-го сообщения пользователя зафиксирован сценарий {scenario}).
{scenario_block}

СПРАВОЧНИК ТИПОВ ПРОБЛЕМ (PROBLEM_KNOWLEDGE_BASE, ключи: {keys}):
{kb_text}

Мета (кратко, для контекста): {json.dumps(um, ensure_ascii=False)[:2500]}
{hypothesis_block}
{work_offer}
"""

        system_message = SystemMessage(content=system_prompt)
        messages_with_system = [system_message] + list(messages)
        llm_response = llm.invoke(messages_with_system)
        content = llm_response.content if hasattr(llm_response, "content") else str(llm_response)

        out: dict[str, Any] = {
            "messages": [AIMessage(content=content)],
            "stage": f"scenario_{scenario}_work",
            "turn_count": turn_count,
        }
        if meta_updates:
            out["user_meta"] = {**um, **meta_updates}
        return out

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
