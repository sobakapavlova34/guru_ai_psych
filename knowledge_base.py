from typing import Dict, List


# типы проблем и их триггерные слова
PROBLEM_KNOWLEDGE_BASE: Dict[str, Dict[str, List[str]]] = {
    "влияние": {
        "trigger_words": [
            "на меня давят родители/учитель/друзья", "говорят/советуют/хотят/решили/заставляют родители / учитель / друзья", 

        ],
        "description": "Влияние на выбор референтных лиц (родителей, друзей, учителей)",
        "litmus_question": "Кто влияет на твой выбор будущей профессии?"
    },
    "поддержка": {
        "trigger_words": [
            "не поддерживают", "не понимают", "не разделяют", "упрекают", "ругают"

        ],
        "description": "Остсутствие поддержки со стороны близких/со стороны карьерного консультанта или службы персонала",
        "litmus_question": "Кто влияет на твой выбор будущей профессии?"
    }
}


def format_problem_kb_for_prompt() -> str:
    lines: List[str] = []
    for key, data in PROBLEM_KNOWLEDGE_BASE.items():
        desc = data.get("description", "")
        litmus = data.get("litmus_question", "")
        lines.append(f"— `{key}`: {desc}")
        if litmus:
            lines.append(f"  (ориентир-вопрос: {litmus})")
    return "\n".join(lines) if lines else "(база пока пуста)"


def find_problem_type(user_message: str) -> tuple[bool, str]:
    if not user_message:
        return False, ""
    
    message_lower = user_message.lower()
    
    for problem_type, data in PROBLEM_KNOWLEDGE_BASE.items():
        trigger_words = data.get("trigger_words", [])
        
        for trigger_word in trigger_words:
            if trigger_word in message_lower:
                return True, problem_type
    
    return False, ""


def get_problem_description(problem_type: str) -> str:

    if problem_type in PROBLEM_KNOWLEDGE_BASE:
        return PROBLEM_KNOWLEDGE_BASE[problem_type].get("description", "")
    return ""


def get_all_problem_types() -> List[str]:
    return list(PROBLEM_KNOWLEDGE_BASE.keys())


def get_trigger_words_for_problem(problem_type: str) -> List[str]:
    if problem_type in PROBLEM_KNOWLEDGE_BASE:
        return PROBLEM_KNOWLEDGE_BASE[problem_type].get("trigger_words", [])
    return []

