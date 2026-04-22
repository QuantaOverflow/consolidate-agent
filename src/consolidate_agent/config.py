from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    dashscope_api_key: str | None = Field(default=None, alias="DASHSCOPE_API_KEY")
    qwen_model: str = Field(default="qwen-flash", alias="QWEN_MODEL")
    dashscope_api_base: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        alias="DASHSCOPE_API_BASE",
    )
    codex_sessions_dir: str = Field(default="~/.codex/sessions", alias="CODEX_SESSIONS_DIR")
    codex_session_index: str = Field(default="~/.codex/session_index.jsonl", alias="CODEX_SESSION_INDEX")
    consolidate_output_dir: str = Field(default="./outputs", alias="CONSOLIDATE_OUTPUT_DIR")
    consolidate_cursor_path: str = Field(default="./outputs/cursor.json", alias="CONSOLIDATE_CURSOR_PATH")

    @property
    def sessions_dir_path(self) -> Path:
        return Path(self.codex_sessions_dir).expanduser()

    @property
    def session_index_path(self) -> Path:
        return Path(self.codex_session_index).expanduser()

    @property
    def output_dir_path(self) -> Path:
        return Path(self.consolidate_output_dir).expanduser()

    @property
    def cursor_path(self) -> Path:
        return Path(self.consolidate_cursor_path).expanduser()
