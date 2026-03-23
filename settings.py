"""Настройки приложения"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Настройки приложения"""
    
    # Bothub API настройки
    BOTHUB_API_KEY: str = ""
    BOTHUB_BASE_URL: str = ""
    BOTHUB_MODEL: str = ""
    BOTHUB_TIMEOUT: int = 30
    
    # Telegram bot
    TELEGRAM_BOT_TOKEN: str = ""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )


# Создаем глобальный экземпляр настроек
settings = Settings()

