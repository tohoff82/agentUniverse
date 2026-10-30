"""Hermetic qualification evidence for the bounded context-engineering APIs."""

from copy import deepcopy

import pytest

from agentuniverse.agent.context.compressor.selective_compressor import (
    SelectiveCompressor,
)
from agentuniverse.agent.context.compressor.truncate_compressor import (
    TruncateCompressor,
)
from agentuniverse.agent.context.context_manager import ContextManager
from agentuniverse.agent.context.context_model import (
    ContextMetadata,
    ContextPriority,
    ContextSegment,
    ContextType,
)
from agentuniverse.agent.context.store.ram_context_store import RamContextStore


def _segment(identifier, content, tokens, *, session_id, priority=ContextPriority.MEDIUM):
    return ContextSegment(
        id=identifier,
        type=ContextType.CONVERSATION,
        priority=priority,
        content=content,
        tokens=tokens,
        session_id=session_id,
        agent_id="qualification-agent",
        task_id="qualification-task",
        parent_id="qualification-parent",
        related_ids=["qualification-related"],
        metadata=ContextMetadata(
            decay_rate=0,
            keywords=[identifier],
            custom={"source": {"id": identifier}},
        ),
    )


def _manager():
    manager = ContextManager(name="qualification-manager", enable_compression=False)
    manager._hot_store = RamContextStore(
        name="qualification-store", max_segments=100, ttl_hours=24
    )
    return manager


def test_ctx_01_session_isolation_for_retrieval_search_and_windows():
    manager = _manager()
    manager.create_context_window("session-a")
    manager.create_context_window("session-b")
    segment_a = manager.add_context(
        "session-a", "shared term private alpha", ContextType.CONVERSATION
    )
    segment_b = manager.add_context(
        "session-b", "shared term private beta", ContextType.CONVERSATION
    )

    assert {item.id for item in manager.get_context("session-a")} == {segment_a.id}
    assert {item.id for item in manager.get_context("session-b")} == {segment_b.id}
    assert {item.id for item in manager.search_context("session-a", "shared term")} == {
        segment_a.id
    }
    assert {item.id for item in manager.search_context("session-b", "shared term")} == {
        segment_b.id
    }
    assert manager.get_context_window("session-a").segment_ids == [segment_a.id]
    assert manager.get_context_window("session-b").segment_ids == [segment_b.id]


def test_ctx_02_budget_enforcement_is_session_scoped():
    manager = _manager()
    window_a = manager.create_context_window(
        "session-a", max_tokens=100, reserved_tokens=20
    )
    window_b = manager.create_context_window(
        "session-b", max_tokens=100, reserved_tokens=20
    )
    protected_b = manager.add_context(
        "session-b", "B" * 160, ContextType.CONVERSATION, ContextPriority.HIGH
    )

    for marker in ("A", "C", "D"):
        manager.add_context(
            "session-a",
            marker * 160,
            ContextType.CONVERSATION,
            ContextPriority.MEDIUM,
        )

    assert window_a.total_tokens <= window_a.calculate_input_tokens()
    assert sum(item.tokens for item in manager.get_context("session-a")) <= 80
    assert window_b.total_tokens <= window_b.calculate_input_tokens()
    assert {item.id for item in manager.get_context("session-b")} == {protected_b.id}

    with pytest.raises(ValueError, match="exceeds the session input budget"):
        manager.add_context(
            "session-a", "X" * 324, ContextType.CONVERSATION, ContextPriority.HIGH
        )
    assert window_a.total_tokens <= window_a.calculate_input_tokens()
    assert {item.id for item in manager.get_context("session-b")} == {protected_b.id}

    candidates = [
        _segment("too-large", "L" * 400, 100, session_id="session-a"),
        _segment("fits", "S" * 40, 10, session_id="session-a"),
    ]
    selected, metrics = SelectiveCompressor().compress(candidates, target_tokens=20)
    assert [item.id for item in selected] == ["fits"]
    assert sum(item.tokens for item in selected) == metrics.compressed_tokens <= 20


def test_ctx_03_truncation_preserves_source_and_result_identity_fields():
    source = _segment(
        "long",
        "semantic-prefix " + "x" * 400,
        110,
        session_id="session-a",
        priority=ContextPriority.HIGH,
    )
    unrelated = _segment("unrelated", "unchanged", 10, session_id="session-a")
    source_before = deepcopy(source.model_dump())
    unrelated_before = deepcopy(unrelated.model_dump())

    result, metrics = TruncateCompressor().compress([source, unrelated], 100)

    by_id = {item.id: item for item in result}
    revision = by_id[source.id]
    assert revision.content.startswith("semantic-prefix ")
    assert revision.id == source.id
    assert revision.session_id == source.session_id
    assert revision.agent_id == source.agent_id
    assert revision.task_id == source.task_id
    assert revision.parent_id == source.parent_id
    assert revision.related_ids == source.related_ids
    assert revision.type == source.type
    assert revision.priority == source.priority
    assert sum(item.tokens for item in result) == metrics.compressed_tokens <= 100
    assert source.model_dump() == source_before
    assert unrelated.model_dump() == unrelated_before


def test_ctx_05_empty_and_filtered_retrieval_do_not_broaden_scope():
    manager = _manager()
    other = manager.add_context(
        "other-session", "globally tempting evidence", ContextType.BACKGROUND
    )

    assert manager.get_context("empty-session") == []
    assert manager.search_context("empty-session", "evidence") == []
    assert manager.get_context(
        "other-session", context_type=ContextType.TOOL_RESULT
    ) == []
    assert manager.search_context(
        "other-session", "evidence", context_type=ContextType.TOOL_RESULT
    ) == []
    assert {item.id for item in manager.get_context("other-session")} == {other.id}
