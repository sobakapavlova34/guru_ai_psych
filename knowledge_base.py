from typing import Dict, List

from knowledge_data.problem_knowledge import PROBLEM_KNOWLEDGE_BASE


def aggregate_kb_trigger_scores(user_meta: dict) -> Dict[str, int]:
    scores: Dict[str, int] = {}
    for block in user_meta.get("kb_trigger_hits") or []:
        if not isinstance(block, dict):
            continue
        for m in block.get("matches") or []:
            if not isinstance(m, dict):
                continue
            pk = (m.get("problem_type") or "").strip()
            if pk:
                scores[pk] = scores.get(pk, 0) + 1
    return scores


def top_problem_keys_by_score(scores: Dict[str, int]) -> List[str]:
    if not scores:
        return []
    max_v = max(scores.values())
    return [k for k, v in scores.items() if v == max_v]


def get_litmus_question_for_problem(problem_key: str) -> str:
    if problem_key not in PROBLEM_KNOWLEDGE_BASE:
        return ""
    return (PROBLEM_KNOWLEDGE_BASE[problem_key].get("litmus_question") or "").strip()


def _trigger_search_fragments(raw: str) -> List[str]:
    raw = (raw or "").strip()
    if not raw:
        return []
    chunks: List[str] = [raw]
    for sep in ("/", "\n", ";"):
        nxt: List[str] = []
        for c in chunks:
            nxt.extend(x.strip() for x in c.split(sep))
        chunks = [x for x in nxt if x]
    # короткие альтернативы через запятую в длинных строках: "а, б, в"
    expanded: List[str] = []
    for c in chunks:
        expanded.append(c)
        if len(c) > 25 and ", " in c:
            expanded.extend(x.strip() for x in c.split(", ") if len(x.strip()) >= 3)
        # отдельные слова из длинных фрагментов («на меня давят родители» → «давят», «родители»)
        if len(c) > 10:
            for w in c.replace(",", " ").split():
                w = w.strip()
                if len(w) >= 4:
                    expanded.append(w)
    # уникальность, порядок; отсекаем слишком короткие (кроме осмысленных 2-симв. «не»)
    seen: set[str] = set()
    out: List[str] = []
    for frag in expanded:
        f = frag.strip()
        if len(f) < 2:
            continue
        if len(f) == 2 and f not in ("не", "ни", "да", "ну"):
            continue
        key = f.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(f)
    return out


def collect_kb_trigger_hits(user_message: str) -> List[Dict[str, str]]:
    if not (user_message or "").strip():
        return []
    message_lower = user_message.lower()
    out: List[Dict[str, str]] = []
    for problem_type, data in PROBLEM_KNOWLEDGE_BASE.items():
        for trigger in data.get("trigger_words", []):
            t = (trigger or "").strip()
            if not t:
                continue
            matched = False
            for frag in _trigger_search_fragments(t):
                if len(frag) < 2:
                    continue
                if frag.lower() in message_lower:
                    out.append(
                        {
                            "problem_type": problem_type,
                            "trigger": frag,
                            "trigger_pattern": t[:200] + ("…" if len(t) > 200 else ""),
                        }
                    )
                    matched = True
                    break
            if matched:
                continue
            if t.lower() in message_lower:
                out.append(
                    {
                        "problem_type": problem_type,
                        "trigger": t,
                        "trigger_pattern": t[:200] + ("…" if len(t) > 200 else ""),
                    }
                )
    return out



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

