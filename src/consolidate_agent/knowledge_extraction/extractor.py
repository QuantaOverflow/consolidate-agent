from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_qwq import ChatQwen
from pydantic import BaseModel, Field

from consolidate_agent.config import Settings
from consolidate_agent.context import ProcessedSession
from consolidate_agent.prompt_loader import load_prompt
from consolidate_agent.types import KnowledgeScope


class KnowledgeItemInput(BaseModel):
    title: str = Field(min_length=3)
    insight: str = Field(min_length=10)
    applicability: str = Field(min_length=10)
    scope: KnowledgeScope


class KnowledgeExtractionOutput(BaseModel):
    items: list[KnowledgeItemInput] = Field(default_factory=list)


class KnowledgeExtractor:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = self._build_model()
        self.structured_model = self.model.with_structured_output(KnowledgeExtractionOutput)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", load_prompt("knowledge_extraction_system.md")),
                ("user", load_prompt("knowledge_extraction_user.md")),
            ],
            template_format="jinja2",
        )

    def _build_model(self) -> ChatQwen:
        if not self.settings.dashscope_api_key:
            raise ValueError(
                "DASHSCOPE_API_KEY is not configured. Fill .env before running real knowledge extraction."
            )
        return ChatQwen(
            model=self.settings.qwen_model,
            api_key=self.settings.dashscope_api_key,
            base_url=self.settings.dashscope_api_base,
            temperature=0,
        )

    def extract(self, session: ProcessedSession) -> list[KnowledgeItemInput]:
        prompt_value = self.prompt.invoke({"session_xml": session.xml})
        result = self.structured_model.invoke(prompt_value)
        if result is None:
            raise ValueError(f"KnowledgeExtractor returned no structured output for session {session.session_id}")
        return list(result.items)
