from __future__ import annotations

from consolidate_agent.chunk import normalized_transcript_hash, split_transcript
from consolidate_agent.types import MessageKind, MessageRole, Transcript, TranscriptMessage, utc_now


def make_transcript(message_count: int = 3, text: str = "hello") -> Transcript:
    return Transcript(
        session_id="session-1",
        messages=[
            TranscriptMessage(
                ref=f"msg_{index:04d}",
                timestamp=utc_now(),
                role=MessageRole.USER,
                kind=MessageKind.MESSAGE,
                text=f"{text}-{index}",
            )
            for index in range(message_count)
        ],
    )


def test_small_transcript_creates_single_chunk() -> None:
    transcript = make_transcript()

    chunks = split_transcript(transcript, max_chunk_chars=10000, overlap_messages=1)

    assert len(chunks) == 1
    assert chunks[0].chunk_id == "session-1:chunk:0000"
    assert [message.ref for message in chunks[0].messages] == ["msg_0000", "msg_0001", "msg_0002"]


def test_large_transcript_splits_with_overlap_and_preserves_refs() -> None:
    transcript = make_transcript(message_count=5, text="x" * 80)

    chunks = split_transcript(transcript, max_chunk_chars=500, overlap_messages=1)

    assert len(chunks) > 1
    assert chunks[0].messages[-1].ref == chunks[1].messages[0].ref
    all_refs = {message.ref for chunk in chunks for message in chunk.messages}
    assert all_refs == {f"msg_{index:04d}" for index in range(5)}
    assert all(chunk.total_chunks == len(chunks) for chunk in chunks)


def test_normalized_transcript_hash_is_stable() -> None:
    first = make_transcript()
    second = make_transcript()
    for left, right in zip(first.messages, second.messages, strict=True):
        right.timestamp = left.timestamp

    assert normalized_transcript_hash(first) == normalized_transcript_hash(second)
