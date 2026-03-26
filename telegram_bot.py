import logging

from typing import Dict

from telegram import (
    Update,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
)
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
)

from langchain_core.messages import HumanMessage, AIMessage

from dialog import create_dialog_graph, DialogState
from settings import settings


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# храним состояния диалога по пользователям
user_states: Dict[int, DialogState] = {}


def build_reset_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(text=" Начать заново", callback_data="reset_dialog")]
    ])


def init_user_state() -> DialogState:
    return {
        "messages": [],
        "conversation_topic": "",
        "stage": "chat",
        "turn_count": 0,
        "user_meta": {},
        "scenario": "",
    }


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    assert update.effective_chat is not None
    chat_id = update.effective_chat.id

    if not hasattr(context.application, "dialog_graph"):
        context.application.dialog_graph = create_dialog_graph()

    dialog_graph = context.application.dialog_graph
    state = init_user_state()

    # приветствие из графа диалога
    result = dialog_graph.invoke(state)
    last_message = result["messages"][-1]
    greeting_text = last_message.content if isinstance(last_message, AIMessage) else "Привет! Чем могу помочь?"

    user_states[chat_id] = {
        **state,
        "messages": result["messages"],
        "turn_count": result.get("turn_count", 1),
        "stage": result.get("stage", "greeting"),
        "user_meta": result.get("user_meta", {}),
        "scenario": result.get("scenario", ""),
    }

    await update.effective_chat.send_message(text=greeting_text)


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    assert update.effective_chat is not None
    chat_id = update.effective_chat.id

    if not hasattr(context.application, "dialog_graph"):
        context.application.dialog_graph = create_dialog_graph()

    state = init_user_state()
    result = context.application.dialog_graph.invoke(state)
    last_message = result["messages"][-1]
    greeting_text = last_message.content if isinstance(last_message, AIMessage) else "Привет! Чем могу помочь?"

    user_states[chat_id] = {
        **state,
        "messages": result["messages"],
        "turn_count": result.get("turn_count", 1),
        "stage": result.get("stage", "greeting"),
        "user_meta": result.get("user_meta", {}),
        "scenario": result.get("scenario", ""),
    }

    await update.effective_chat.send_message(text="🔄 Диалог очищен.\n\n" + greeting_text)


async def reset_dialog_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    assert query is not None
    await query.answer("Диалог очищен")

    await reset(update, context)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    assert update.effective_chat is not None
    chat_id = update.effective_chat.id
    text = update.message.text if update.message else ""

    if not text:
        return

    if not hasattr(context.application, "dialog_graph"):
        context.application.dialog_graph = create_dialog_graph()

    dialog_graph = context.application.dialog_graph

    state = user_states.get(chat_id)
    if state is None:
        await start(update, context)
        return

    # добавляем сообщение пользователя в историю
    state["messages"].append(HumanMessage(content=text))

    result = dialog_graph.invoke(state)
    last_message = result["messages"][-1]

    if isinstance(last_message, AIMessage):
        reply_text = last_message.content
    else:
        reply_text = "Продолжим."

    user_states[chat_id] = {
        **state,
        **{
            "messages": result["messages"],
            "stage": result.get("stage", state.get("stage", "chat")),
            "turn_count": result.get("turn_count", state.get("turn_count", 0)),
            "tell_about_yourself_pending": result.get("tell_about_yourself_pending", 0),
            "user_meta": result.get("user_meta", state.get("user_meta", {})),
            "scenario": result.get("scenario", state.get("scenario", "")),
        },
    }

    await update.effective_chat.send_message(
        reply_text,
        reply_markup=build_reset_keyboard()
    )


def main() -> None:
    if not getattr(settings, "TELEGRAM_BOT_TOKEN", None):
        raise RuntimeError("TELEGRAM_BOT_TOKEN не задан в .env")

    application = (
        ApplicationBuilder()
        .token(settings.TELEGRAM_BOT_TOKEN)
        .build()
    )

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("reset", reset))
    application.add_handler(CommandHandler("clear", reset))  # альтернативная команда
    application.add_handler(
        CallbackQueryHandler(reset_dialog_callback, pattern=r"^reset_dialog$")
    )
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
    )

    application.run_polling()


if __name__ == "__main__":
    main()


