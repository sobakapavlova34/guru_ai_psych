from typing import TypedDict, Annotated, Sequence, Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from llm.bothub_client import bothub_client
import operator

import traceback

BASE_SYSTEM_PROMPT = """Ты как друг — лёгкий, тёплый, без официальности. Общайся на "ты".
Можно просто выговориться, без давления. Человек может не знать своей проблемы — это ок."""

TELL_ABOUT_YOURSELF_QUESTIONS = [
    "Сколько тебе лет?",
    "Где живёшь?",
    "Где учишься или учился?",
    "Расскажи про семью",
]

OFF_TOPIC_KEYWORDS = ("проблем", "здоровь", "болит", "цел", "привело", "пришёл", "пришла", "беспоко", "тревож", "депресс", "стресс")

class DialogState(TypedDict, total=False):

    messages: Annotated[Sequence[BaseMessage], operator.add]
    conversation_topic: str
    stage: str
    turn_count: int
    tell_about_yourself_pending: int  # при офф-топик: не считать последнее сообщение для маршрутизации


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
        # просто передаем состояние дальше, решение принимается через условные рёбра
        return state
    
    def _get_effective_count(state: DialogState) -> int:
        messages = state.get("messages", [])
        user_msg_count = sum(1 for m in messages if isinstance(m, HumanMessage))
        pending = state.get("tell_about_yourself_pending", 0)
        return user_msg_count - pending

    def should_route(state: DialogState) -> Literal["greeting", "dialog", "ended"]:
        """Маршрут: greeting (0) -> dialog (1-7) -> ended (8+)."""
        effective_count = _get_effective_count(state)
        if effective_count == 0:
            return "greeting"
        if effective_count >= 8:
            return "ended"
        return "dialog"
    
    def greeting_node(state: DialogState) -> DialogState:
        turn_count = state.get("turn_count", 0)
        turn_count += 1
        
        greeting_text = (
            """Привет! Я тут, чтобы поболтать и помочь разобраться, если что-то гложет.
            Устраивайся поудобнее. Как тебя зовут?
            """
        )
        
        response = AIMessage(content=greeting_text)
        
        return {
            "messages": [response],
            "stage": "setting_up_a_contact",
            "turn_count": turn_count
        }
    
    def _get_stage_instructions(effective_count: int, flag_barrier: bool) -> tuple[str, str | None, str]:
        if effective_count == 1:
            q = "Ты что-нибудь слышал о том, как я работаю? Или просто хочешь поболтать?"
            return (
                f"""ЭТАП: установление контакта. Твоя задача — ТОЛЬКО представиться.
- Пользователь написал имя — представься зеркально (Елена→Мария, Дэн→Кэт).
- В конце задай ТОЧНО этот вопрос: «{q}»
ЗАПРЕЩЕНО: рассказывать о консультировании; спрашивать о проблеме, здоровье, возрасте, семье.""",
                q,
                "setting_up_a_contact",
            )
        if effective_count == 2:
            return (
                """ЭТАП: знакомство с форматом работы. Расскажи коротко, как ты работаешь:
- Помогаю разобраться в ситуации, найти выход. Можно просто выговориться.
- Всё остаётся между нами. (Угроза жизни — предупрежу.)
- Иногда лучше к живому специалисту — подскажу.
- Пиши развёрнуто, как душе угодно.
В конце спроси: "Окей, давай познакомимся поближе? Напиши что-нибудь."
ЗАПРЕЩЕНО: спрашивать о проблеме, здоровье, семье, "что привело".""",
                "Окей, давай познакомимся поближе? Напиши что-нибудь.",
                "knowledge_about_psychological_counseling",
            )
        if 1 <= effective_count <= 6: # надо сделать это основным условием для этапа дружеской беседы, выводим на тригерные слова
            index = effective_count - 3
            question = TELL_ABOUT_YOURSELF_QUESTIONS[index]
            return (
                f"""ЭТАП: знакомство. Можешь коротко отреагировать на ответ.
В конце ОБЯЗАТЕЛЬНО задай ТОЧНО этот вопрос: «{question}»
Если пользователь ушёл в сторону (о проблеме, здоровье) — мягко верни к вопросу.""",
                question,
                "tell_about_yourself",
            )
        if effective_count == 7:
            q = "Что привело тебя сюда? О чём хочется поговорить?" # предполагаем проблему
            return (
                f"""ЭТАП: цель визита. Можешь отреагировать на ответ.
В конце ОБЯЗАТЕЛЬНО задай ТОЧНО этот вопрос: «{q}»""",
                q,
                "users_goal",
            )
        if effective_count == 8:
            return (f"""""", flag_barrier) # проблема (не)установлена
        return ("", None, "dialog")
    
    def barrier_detected(effective_count: int, flag: bool) -> tuple[str, str | None]:
        if effective_count == 1 and flag == True: 
            return (f"""подтверждение барьера; фидбэк - истории других; предлагаем инстурменты""")
        else:
            return ()


    def dialog_node(state: DialogState) -> DialogState:
        messages = state["messages"]
        turn_count = state.get("turn_count", 0) + 1
        pending = state.get("setting_up_a_contact", 0)
        effective_count = _get_effective_count(state)

        index = effective_count - 3
        is_off_topic = False
        if 3 <= effective_count <= 6 and index > 0:
            last_user_text = ""
            for m in reversed(messages):
                if isinstance(m, HumanMessage):
                    last_user_text = (m.content or "").lower()
                    break
            if any(kw in last_user_text for kw in OFF_TOPIC_KEYWORDS):
                is_off_topic = True
                effective_count -= 1

        stage_instructions, fixed_question, stage_name = _get_stage_instructions(effective_count)
        system_prompt = f"{BASE_SYSTEM_PROMPT}\n\n{stage_instructions}"

        system_message = SystemMessage(content=system_prompt)
        messages_with_system = [system_message] + list(messages)
        
        llm_response = llm.invoke(messages_with_system)
        content = llm_response.content if hasattr(llm_response, "content") else str(llm_response)
        if fixed_question and fixed_question not in content:
            content = f"{content.strip()}\n\n{fixed_question}" if content.strip() else fixed_question
        response = AIMessage(content=content)
        
        result: dict = {"messages": [response], "stage": stage_name, "turn_count": turn_count}
        result["tell_about_yourself_pending"] = pending + 1 if is_off_topic else 0
        return result

    def ended_node(state: DialogState) -> DialogState:
        turn_count = state.get("turn_count", 0) + 1
        closing = "На сегодня всё. Спасибо, что поделился. Если захочешь поговорить снова — я здесь."
        return {
            "messages": [AIMessage(content=closing)],
            "stage": "ended",
            "turn_count": turn_count,
        }

    # граф состояний
    # StateGraph - это структура, которая управляет потоком выполнения
    # автоматически передает состояние между узлами; узлы взаимодействуют посредством чтения и записи в общее состояние.
    graph = StateGraph(DialogState)
    
    
    graph.add_node("router", router_node)
    graph.add_node("greeting", greeting_node)
    graph.add_node("dialog", dialog_node)
    graph.add_node("ended", ended_node)
    
    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        should_route,
        {
            "greeting": "greeting",
            "dialog": "dialog",
            "ended": "ended",
        }
    )
    
    graph.add_edge("greeting", END)
    graph.add_edge("dialog", END)
    graph.add_edge("ended", END)

    return graph.compile() 


def run_dialog(
    top_p: float = 1,
    max_completion_tokens: int = 2000,
    temperature: float | None = 0.2,
):
    
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
                }
                first_run = False
            except Exception as e:
                import traceback
                traceback.print_exc(e)
                break
        
        user_input = input("Вы: ").strip()
        
        if not user_input:
            continue
        
        state["messages"].append(HumanMessage(content=user_input))

        try:
            result = dialog_graph.invoke(state)

            last_message = result["messages"][-1]
            current_turn = result.get("turn_count", 0)
            
            if isinstance(last_message, AIMessage):
                print(f"Бот: {last_message.content}")
                print(f"[Шаг {current_turn}/{MAX_TURNS}]\n") 
            
            if current_turn >= MAX_TURNS:
                print(" До свидания!")
                break
            
            
            # обновляем состояние для следующей итерации и  сохраняем все поля из предыдущего состояния
            state = {
                "messages": result["messages"],
                "conversation_topic": state.get("conversation_topic", ""),
                "stage": result.get("stage", state.get("stage", "chat")),
                "turn_count": result.get("turn_count", 0),
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            break


if __name__ == "__main__":
    run_dialog(
        top_p=1,
        max_completion_tokens=2000,
        temperature=0.2,
    )
