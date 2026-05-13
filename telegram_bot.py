import logging
import traceback

from typing import Any, Dict

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

from data_store import (
    get_reaction_options,
    init_storage,
    seed_barriers,
    save_bot_message,
    save_reaction,
)
from dialog import create_dialog_graph, DialogState
from settings import settings


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DIALOG_GRAPH_KEY = "dialog_graph"

# храним состояния диалога по пользователям
user_states: Dict[int, DialogState] = {}


def get_dialog_graph(context: ContextTypes.DEFAULT_TYPE) -> Any:
    """Граф создаётся один раз и хранится в application.bot_data (нельзя добавлять новые атрибуты в Application)."""
    bd = context.application.bot_data
    if DIALOG_GRAPH_KEY not in bd:
        bd[DIALOG_GRAPH_KEY] = create_dialog_graph()
    return bd[DIALOG_GRAPH_KEY]


def build_reset_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(text=" Начать заново", callback_data="reset_dialog")]
    ])


def build_reply_keyboard() -> InlineKeyboardMarkup:
    reaction_buttons = [
        InlineKeyboardButton(text=row["label"], callback_data=f"react:{row['code']}")
        for row in get_reaction_options()
    ]
    rows: list[list[InlineKeyboardButton]] = []
    if reaction_buttons:
        rows.append(reaction_buttons)
    rows.extend(build_reset_keyboard().inline_keyboard)
    return InlineKeyboardMarkup(rows)


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

    dialog_graph = get_dialog_graph(context)
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

    sent = await update.effective_chat.send_message(
        text=greeting_text,
        reply_markup=build_reply_keyboard(),
    )
    save_bot_message(
        chat_id=chat_id,
        user_id=update.effective_user.id if update.effective_user else None,
        message_id=sent.message_id,
        text=greeting_text,
        stage=user_states[chat_id].get("stage", ""),
        scenario=user_states[chat_id].get("scenario", ""),
    )


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    assert update.effective_chat is not None
    chat_id = update.effective_chat.id

    state = init_user_state()
    result = get_dialog_graph(context).invoke(state)
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

    text_out = "🔄 Диалог очищен.\n\n" + greeting_text
    sent = await update.effective_chat.send_message(
        text=text_out,
        reply_markup=build_reply_keyboard(),
    )
    save_bot_message(
        chat_id=chat_id,
        user_id=update.effective_user.id if update.effective_user else None,
        message_id=sent.message_id,
        text=text_out,
        stage=user_states[chat_id].get("stage", ""),
        scenario=user_states[chat_id].get("scenario", ""),
    )


async def reset_dialog_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    assert query is not None
    await query.answer("Диалог очищен")

    await reset(update, context)


async def reaction_callback(update: Update, _context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if query is None or query.data is None or not query.data.startswith("react:"):
        return
    if query.message is None or query.from_user is None:
        await query.answer("Не удалось сохранить")
        return
    reaction_code = query.data.split(":", 1)[1].strip()
    ok = save_reaction(
        chat_id=query.message.chat_id,
        user_id=query.from_user.id,
        message_id=query.message.message_id,
        reaction_code=reaction_code,
    )
    await query.answer("Реакция сохранена" if ok else "Ошибка сохранения")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    assert update.effective_chat is not None
    chat_id = update.effective_chat.id
    text = update.message.text if update.message else ""

    if not text:
        return

    dialog_graph = get_dialog_graph(context)

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

    sent = await update.effective_chat.send_message(
        reply_text,
        reply_markup=build_reply_keyboard(),
    )
    save_bot_message(
        chat_id=chat_id,
        user_id=update.effective_user.id if update.effective_user else None,
        message_id=sent.message_id,
        text=reply_text,
        stage=user_states[chat_id].get("stage", ""),
        scenario=user_states[chat_id].get("scenario", ""),
    )


async def telegram_error(_update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Сетевые таймауты к api.telegram.org и прочие ошибки — в лог, без сообщения «No error handlers»."""
    err = context.error
    if err is not None:
        tb = getattr(err, "__traceback__", None)
        lines = traceback.format_exception(type(err), err, tb)
        logger.error("Ошибка в обработчике Telegram:\n%s", "".join(lines))


def main() -> None:
    if not getattr(settings, "TELEGRAM_BOT_TOKEN", None):
        raise RuntimeError("TELEGRAM_BOT_TOKEN не задан в .env")

    init_storage()
    loaded_rows = seed_barriers()
    logger.info("barrier rows loaded: %s", loaded_rows)

    application = (
        ApplicationBuilder()
        .token(settings.TELEGRAM_BOT_TOKEN)
        .connect_timeout(30.0)
        .read_timeout(30.0)
        .write_timeout(30.0)
        .build()
    )

    application.add_error_handler(telegram_error)

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("reset", reset))
    application.add_handler(CommandHandler("clear", reset))  # альтернативная команда
    application.add_handler(
        CallbackQueryHandler(reset_dialog_callback, pattern=r"^reset_dialog$")
    )
    application.add_handler(
        CallbackQueryHandler(reaction_callback, pattern=r"^react:")
    )
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
    )

    application.run_polling()


if __name__ == "__main__":
    main()


