from __future__ import annotations

import hashlib
import json

from consolidate_agent.types import Transcript, TranscriptChunk, TranscriptMessage


def normalized_transcript_hash(transcript: Transcript) -> str:
    payload = transcript.model_dump(mode="json")
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def split_transcript(
    transcript: Transcript,
    max_chunk_chars: int = 30000,
    overlap_messages: int = 5,
) -> list[TranscriptChunk]:
    if max_chunk_chars <= 0:
        raise ValueError("max_chunk_chars must be greater than 0")
    if overlap_messages < 0:
        raise ValueError("overlap_messages must be greater than or equal to 0")

    messages = transcript.messages
    if not messages:
        return [_build_chunk(transcript.session_id, 0, 1, [])]

    chunks: list[list[TranscriptMessage]] = []
    current: list[TranscriptMessage] = []
    current_size = 0

    for message in messages:
        message_size = _message_size(message)
        if current and current_size + message_size > max_chunk_chars:
            chunks.append(current)
            overlap_count = min(overlap_messages, max(len(current) - 1, 0))
            overlap = current[-overlap_count:] if overlap_count else []
            current = list(overlap)
            current_size = sum(_message_size(item) for item in current)
        current.append(message)
        current_size += message_size

    if current:
        chunks.append(current)

    total = len(chunks)
    return [_build_chunk(transcript.session_id, index, total, chunk_messages) for index, chunk_messages in enumerate(chunks)]


def _build_chunk(
    session_id: str,
    chunk_index: int,
    total_chunks: int,
    messages: list[TranscriptMessage],
) -> TranscriptChunk:
    return TranscriptChunk(
        session_id=session_id,
        chunk_id=f"{session_id}:chunk:{chunk_index:04d}",
        chunk_index=chunk_index,
        total_chunks=total_chunks,
        messages=messages,
        char_count=sum(_message_size(message) for message in messages),
    )


def _message_size(message: TranscriptMessage) -> int:
    payload = message.model_dump(mode="json")
    return len(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
