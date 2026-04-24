from __future__ import annotations

import json
import traceback
from typing import Any

import gradio as gr
from langchain_core.messages import AIMessage, HumanMessage

from dialog import DialogState, _get_effective_count, create_dialog_graph, logger

_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = create_dialog_graph()
    return _graph


def build_initial_state() -> DialogState:
    return {
        "messages": [],
        "conversation_topic": "",
        "stage": "chat",
        "turn_count": 0,
        "user_meta": {},
        "scenario": "",
    }


def merge_from_result(prev: DialogState, result: DialogState) -> DialogState:
    return {
        "messages": result["messages"],
        "conversation_topic": prev.get("conversation_topic", ""),
        "stage": result.get("stage", "chat"),
        "turn_count": result.get("turn_count", 0),
        "tell_about_yourself_pending": result.get("tell_about_yourself_pending", 0),
        "user_meta": result.get("user_meta", {}),
        "scenario": result.get("scenario", ""),
    }


def lc_messages_to_chat(messages: list) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for m in messages:
        if isinstance(m, HumanMessage):
            out.append({"role": "user", "content": m.content or ""})
        elif isinstance(m, AIMessage):
            out.append({"role": "assistant", "content": m.content or ""})
    return out


def format_flags_md(state: DialogState) -> str:
    meta = state.get("user_meta") or {}
    finalized = bool(meta.get("intro_flags_finalized")) if isinstance(meta, dict) else False
    flags = (meta.get("flags") or {}) if isinstance(meta, dict) else {}
    mp = flags.get("mentions_problem", False)
    pu = flags.get("problem_unclear", False)
    cc = flags.get("casual_chat", False)
    rationale = (meta.get("flags_rationale") or "").strip() if isinstance(meta, dict) else ""
    sc = state.get("scenario") or "—"
    st = state.get("stage") or "—"
    tc = state.get("turn_count", 0)
    pending = (
        "\n_Флаги `mentions_problem` / `problem_unclear` / `casual_chat` выставляются **один раз** "
        "после завершения 5 вводных шагов по всей накопленной метадате._\n"
        if not finalized
        else ""
    )
    rat_block = f"\n**Обоснование:** {rationale}\n" if rationale else ""
    hyp_k = (meta.get("hypothesized_problem_key") or "").strip() if isinstance(meta, dict) else ""
    hyp_q = (meta.get("hypothesis_litmus_question") or "").strip() if isinstance(meta, dict) else ""
    hyp_rs = (meta.get("hypothesis_reason") or "").strip() if isinstance(meta, dict) else ""
    hyp_block = ""
    if hyp_k or hyp_q:
        hyp_block = (
            f"\n### Гипотеза проблемы (после шага 5)\n"
            f"- **Ключ из базы:** `{hyp_k or '—'}`\n"
            f"- **Уточняющий вопрос:** {hyp_q or '—'}\n"
        )
        if hyp_rs:
            hyp_block += f"- **Как выбрано:** {hyp_rs}\n"
    return f"""### Состояние диалога
- **Этап / stage:** `{st}`
- **Сценарий (после шага 6):** `{sc}`
- **turn_count:** `{tc}`
- **Флаги intro финализированы:** `{'да' if finalized else 'нет'}`{pending}{rat_block}{hyp_block}
### Флаги (итог по 5 вводным шагам)
| Сигнал | Значение |
|--------|----------|
| mentions_problem | {'✓' if finalized and mp else ('—' if finalized else '…')} |
| problem_unclear | {'✓' if finalized and pu else ('—' if finalized else '…')} |
| casual_chat | {'✓' if finalized and cc else ('—' if finalized else '…')} |
"""


def format_insights_md(state: DialogState) -> str:
    meta = state.get("user_meta") or {}
    insights = meta.get("step_insights") if isinstance(meta, dict) else None
    kb_hits = meta.get("kb_trigger_hits") if isinstance(meta, dict) else None

    parts: list[str] = []

    if insights:
        lines = ["### Инсайты по шагам (LLM)", ""]
        for item in insights:
            if not isinstance(item, dict):
                continue
            step = item.get("intro_step", "?")
            interp = item.get("interpretation", "")
            note = item.get("portrait_note", "")
            lines.append(f"**Шаг {step}**")
            if interp:
                lines.append(f"- *Интерпретация:* {interp}")
            if note:
                lines.append(f"- *Портрет:* {note}")
            lines.append("")
        parts.append("\n".join(lines))
    else:
        parts.append(
            "### Инсайты по шагам\n_Пока нет (после первых реплик появятся интерпретации LLM)._"
        )

    if kb_hits:
        tw = ["### Совпадения с `trigger_words` (база, с intro_step ≥ 3)", ""]
        for block in kb_hits:
            if not isinstance(block, dict):
                continue
            step = block.get("intro_step", "?")
            matches = block.get("matches") or []
            tw.append(f"**Шаг {step}**")
            if not matches:
                tw.append("_Совпадений с фрагментами триггеров нет (реплика проверена)._")
            for m in matches:
                if not isinstance(m, dict):
                    continue
                pk = m.get("problem_type", "")
                tr = m.get("trigger", "")
                short = tr if len(tr) <= 120 else tr[:117] + "…"
                tw.append(f"- `{pk}` ← «{short}»")
            tw.append("")
        parts.append("\n".join(tw))
    elif insights:
        parts.append(
            "### Совпадения с `trigger_words`\n"
            "_Проверка ещё не запускалась (триггеры ищутся с 3-го пользовательского шага ввода)._"
        )

    return "\n\n".join(parts) if parts else (
        "### Инсайты по шагам\n_Пока нет._\n\n### trigger_words\n_С 3-го вводного шага._"
    )


def format_meta_json(state: DialogState) -> str:
    meta = state.get("user_meta") or {}
    try:
        return json.dumps(meta, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        return str(meta)


def init_chat() -> tuple[Any, ...]:
    graph = get_graph()
    state = build_initial_state()
    try:
        result = graph.invoke(state)
        state = merge_from_result(state, result)
        history = lc_messages_to_chat(list(state["messages"]))
        return (
            history,
            state,
            format_flags_md(state),
            format_insights_md(state),
            format_meta_json(state),
        )
    except Exception as e:
        logger.exception("init_chat")
        err = f"Ошибка приветствия: {e}\n{traceback.format_exc()}"
        return [], build_initial_state(), f"### Ошибка\n{err}", "", "{}"


def respond(
    message: str,
    state: DialogState | None,
) -> tuple[Any, ...]:
    if state is None:
        state = build_initial_state()
    text = (message or "").strip()
    if not text:
        history = lc_messages_to_chat(list(state.get("messages", [])))
        return (
            history,
            state,
            format_flags_md(state),
            format_insights_md(state),
            format_meta_json(state),
        )

    graph = get_graph()
    try:
        msgs = list(state["messages"])
        msgs.append(HumanMessage(content=text))
        state = {**state, "messages": msgs}

        if _get_effective_count(state) == 6:
            logger.info(
                "Шаг 6: переход к классификации - готов определить фокус диалога."
            )

        result = graph.invoke(state)
        state = merge_from_result(state, result)
        history = lc_messages_to_chat(list(state["messages"]))

        if str(result.get("stage", "")).startswith("classified_scenario_"):
            note = (state.get("user_meta") or {}).get("classification", {}).get("note", "")
            logger.info(
                "Классификация после шага 6: сценарий=%s.%s",
                state.get("scenario") or "?",
                f" {note}" if note else "",
            )

        return (
            history,
            state,
            format_flags_md(state),
            format_insights_md(state),
            format_meta_json(state),
        )
    except Exception as e:
        logger.exception("respond")
        err = f"Ошибка: {e}"
        history = lc_messages_to_chat(list(state.get("messages", [])))
        history.append({"role": "assistant", "content": err})
        return (
            history,
            state,
            format_flags_md(state),
            format_insights_md(state),
            format_meta_json(state),
        )


def reset_chat() -> tuple[Any, ...]:
    return init_chat()


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Guru AI — диалог") as demo:
        gr.Markdown(
            "## Диалог с автопсихологом\n"
            "Слева — чат. Справа — **метаинформация**, которую бот накапливает (флаги, инсайты LLM, полный JSON)."
        )
        state = gr.State(value=None)

        with gr.Row():
            with gr.Column(scale=3, min_width=400):
                chatbot = gr.Chatbot(
                    label="Чат",
                    height=480,
                    buttons=["copy"],
                )
                with gr.Row():
                    msg = gr.Textbox(
                        label="Ваше сообщение",
                        placeholder="Напишите сообщение и нажмите Enter или «Отправить»",
                        lines=2,
                        scale=4,
                    )
                    send_btn = gr.Button("Отправить", variant="primary", scale=1)
                with gr.Row():
                    reset_btn = gr.Button("Начать заново", variant="secondary")

            with gr.Column(scale=2, min_width=280):
                gr.Markdown("### Панель метаданных")
                flags_panel = gr.Markdown(value="Загрузка…")
                insights_panel = gr.Markdown(value="")
                gr.Markdown("#### Полный `user_meta` (JSON)")
                json_panel = gr.Code(
                    label="user_meta",
                    language="json",
                    lines=18,
                    interactive=False,
                )

        demo.load(
            fn=init_chat,
            inputs=None,
            outputs=[chatbot, state, flags_panel, insights_panel, json_panel],
        )

        send_btn.click(
            fn=respond,
            inputs=[msg, state],
            outputs=[chatbot, state, flags_panel, insights_panel, json_panel],
        ).then(lambda: "", outputs=[msg])

        msg.submit(
            fn=respond,
            inputs=[msg, state],
            outputs=[chatbot, state, flags_panel, insights_panel, json_panel],
        ).then(lambda: "", outputs=[msg])

        reset_btn.click(
            fn=reset_chat,
            inputs=None,
            outputs=[chatbot, state, flags_panel, insights_panel, json_panel],
        )

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
    )
