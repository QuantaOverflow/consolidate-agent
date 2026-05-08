from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import logging
import re
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_qwq import ChatQwen
from pydantic import BaseModel, Field

from consolidate_agent.config import Settings
from consolidate_agent.types import KnowledgeRecord

logger = logging.getLogger(__name__)

_DEDUP_BATCH_SIZE = 100


class TagOutput(BaseModel):
    tags: list[str] = Field(min_length=1, max_length=5)


class _TagMap(BaseModel):
    original: str
    canonical: str


class _TagDeduplicationOutput(BaseModel):
    mappings: list[_TagMap] = Field(default_factory=list)


class KnowledgeTagExtractor:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.structured_model = self._build_model()
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Extract 1-5 normalized technical or conceptual tags. "
                    "Tags must be canonical nouns, lowercase, and not natural language phrases. "
                    "Prefer concrete technologies or concepts such as pytest, langgraph, mysql, volcengine, sse.",
                ),
                (
                    "user",
                    "Title: {{ title }}\n\nInsight:\n{{ insight }}\n\nApplicability:\n{{ applicability }}",
                ),
            ],
            template_format="jinja2",
        )

    def _build_model(self):
        if not self.settings.dashscope_api_key:
            logger.warning("DASHSCOPE_API_KEY is not configured; knowledge tag extraction will return empty tags")
            return None
        return ChatQwen(
            model="qwen-plus",
            api_key=self.settings.dashscope_api_key,
            base_url=self.settings.dashscope_api_base,
            temperature=0,
        ).with_structured_output(TagOutput)

    def extract_tags(self, record: KnowledgeRecord) -> list[str]:
        if self.structured_model is None:
            return []
        prompt_value = self.prompt.invoke(
            {
                "title": record.title,
                "insight": record.insight,
                "applicability": record.applicability,
            }
        )
        for attempt in range(1, 4):
            try:
                result = self.structured_model.invoke(prompt_value)
            except Exception as exc:  # noqa: BLE001 - structured output failures should not abort the batch
                logger.warning(
                    "Knowledge tag extraction failed for %s on attempt %s: %s",
                    record.id,
                    attempt,
                    exc,
                )
                continue
            if result is not None:
                tags = _normalize_tags(result.tags)
                if tags:
                    return tags
                logger.warning("Knowledge tag extraction produced no valid tags for %s on attempt %s", record.id, attempt)
                continue
            logger.warning("Knowledge tag extraction returned no output for %s on attempt %s", record.id, attempt)
        logger.warning("Knowledge tag extraction exhausted retries for %s", record.id)
        return []

    def extract_tags_batch(self, records: list[KnowledgeRecord]) -> dict[str, list[str]]:
        with ThreadPoolExecutor(max_workers=10) as executor:
            tags_by_record = executor.map(self.extract_tags, records)
            return {record.id: tags for record, tags in zip(records, tags_by_record, strict=True)}


def deduplicate_tags(tags: list[str], settings: Settings) -> dict[str, str]:
    """Return a merge map {old_tag: canonical_tag} for semantically overlapping tags.

    Tags not present in the map are unchanged.
    """
    if not tags:
        return {}
    model = _build_dedup_model(settings)
    if model is None:
        return {}

    original_tags = _unique_normalized_tags(tags)
    if not original_tags:
        return {}

    original_to_local: dict[str, str] = {}
    for start in range(0, len(original_tags), _DEDUP_BATCH_SIZE):
        batch = original_tags[start : start + _DEDUP_BATCH_SIZE]
        local_map = _dedup_batch(batch, model)
        for tag in batch:
            original_to_local[tag] = local_map.get(tag, tag)

    local_canonicals = _unique_normalized_tags(original_to_local.values())
    global_map = _dedup_canonicals(local_canonicals, model)

    final_map: dict[str, str] = {}
    for original, local in original_to_local.items():
        canonical = global_map.get(local, local)
        if original != canonical:
            final_map[original] = canonical
    return final_map


def _build_dedup_model(settings: Settings) -> Any | None:
    if not settings.dashscope_api_key:
        return None
    return ChatQwen(
        model="qwen-max",
        api_key=settings.dashscope_api_key,
        base_url=settings.dashscope_api_base,
        temperature=0,
    ).with_structured_output(_TagDeduplicationOutput)


def _dedup_canonicals(tags: list[str], model) -> dict[str, str]:
    if len(tags) <= _DEDUP_BATCH_SIZE:
        return _dedup_batch(tags, model)

    local_to_reduced: dict[str, str] = {}
    for start in range(0, len(tags), _DEDUP_BATCH_SIZE):
        batch = tags[start : start + _DEDUP_BATCH_SIZE]
        batch_map = _dedup_batch(batch, model)
        for tag in batch:
            local_to_reduced[tag] = batch_map.get(tag, tag)

    reduced_canonicals = _unique_normalized_tags(local_to_reduced.values())
    if reduced_canonicals == tags:
        return {tag: reduced for tag, reduced in local_to_reduced.items() if tag != reduced}

    reduced_to_global = _dedup_canonicals(reduced_canonicals, model)
    final_map: dict[str, str] = {}
    for tag, reduced in local_to_reduced.items():
        canonical = reduced_to_global.get(reduced, reduced)
        if tag != canonical:
            final_map[tag] = canonical
    return final_map


def _dedup_batch(tags: list[str], model) -> dict[str, str]:
    """Deduplicate a single tag batch with the configured structured LLM."""
    if not tags or model is None:
        return {}
    prompt = (
        "You are normalizing a set of knowledge tags. "
        "For each tag that has a semantically equivalent or more general form in the list, "
        "output a mapping from the redundant tag to the canonical one. "
        "Prefer shorter, more general forms (e.g. 'llm' over 'llm-prompting', 'json' over 'json-generation'). "
        "Only output mappings where the original and canonical are different. "
        "Tags with no equivalent should not appear in the output.\n\n"
        "Tags:\n"
        + "\n".join(f"- {t}" for t in sorted(tags))
        + "\n\nReturn a flat mappings list of {original, canonical} pairs."
    )
    for attempt in range(1, 4):
        try:
            result = model.invoke(prompt)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Tag deduplication failed on attempt %s: %s", attempt, exc)
            continue
        if result is None:
            logger.warning("Tag deduplication returned None on attempt %s", attempt)
            continue
        merge_map: dict[str, str] = {}
        for mapping in result.mappings:
            original = _normalize_tag(mapping.original)
            canonical = _normalize_tag(mapping.canonical)
            if original and canonical and original != canonical:
                merge_map[original] = canonical
        return merge_map
    logger.warning("Tag deduplication exhausted retries, returning empty map")
    return {}


def _unique_normalized_tags(tags) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        value = _normalize_tag(tag)
        if not value or value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    return normalized


def _normalize_tag(tag: str) -> str:
    value = re.sub(r"\s+", "-", tag.strip().lower())
    return re.sub(r"[^a-z0-9_.+-]", "", value)


def _normalize_tags(tags: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        value = re.sub(r"\s+", "-", tag.strip().lower())
        value = re.sub(r"[^a-z0-9_.+-]", "", value)
        if not value or value in seen:
            continue
        seen.add(value)
        normalized.append(value)
        if len(normalized) == 5:
            break
    return normalized
