from langchain_openai import ChatOpenAI
from pydantic import SecretStr
import sys

from settings import settings 

def bothub_client(
    top_p: float = 1,
    max_completion_tokens: int = 700,
    temperature: float | None = 0.2,
) -> ChatOpenAI:
    return ChatOpenAI(
        api_key=SecretStr(settings.BOTHUB_API_KEY),
        base_url=settings.BOTHUB_BASE_URL,
        model=settings.BOTHUB_MODEL,
        top_p=top_p,
        max_completion_tokens=max_completion_tokens,
        temperature=temperature,
    )
