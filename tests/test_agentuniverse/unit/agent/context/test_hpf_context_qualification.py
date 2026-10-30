"""Hermetic qualification evidence for the bounded context-engineering APIs."""

from copy import deepcopy
from unittest.mock import MagicMock

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
from agentuniverse.agent.context.store.chroma_context_store import ChromaContextStore
from agentuniverse.agent.context.store.redis_context_store import RedisContextStore


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


def test_oversized_unseen_session_is_rejected_without_registration_or_storage():
    manager = _manager()
    storage_before = deepcopy(manager._hot_store._storage)

    with pytest.raises(ValueError, match="exceeds the session input budget"):
        manager.add_context(
            "unseen-session",
            "X" * 28004,
            ContextType.CONVERSATION,
        )

    assert manager.get_context_window("unseen-session") is None
    assert manager._hot_store._storage == storage_before


def test_compression_targets_capacity_remaining_after_incoming_segment():
    manager = ContextManager(name="qualification-manager", enable_compression=True)
    manager._hot_store = RamContextStore(
        name="qualification-store", max_segments=100, ttl_hours=24
    )
    window = manager.create_context_window(
        "session-a", max_tokens=100, reserved_tokens=20
    )
    protected = manager.add_context(
        "session-a",
        "P" * 240,
        ContextType.SYSTEM,
        ContextPriority.CRITICAL,
    )

    class NonCompressingCompressor:
        def __init__(self):
            self.targets = []

        def compress(self, segments, target_tokens, **kwargs):
            self.targets.append(target_tokens)
            return list(segments), object()

    compressor = NonCompressingCompressor()
    manager._compressor = compressor

    with pytest.raises(ValueError, match="Unable to create sufficient context capacity"):
        manager.add_context(
            "session-a",
            "N" * 120,
            ContextType.CONVERSATION,
        )

    assert compressor.targets == [50]
    assert window.total_tokens == 60
    assert window.segment_ids == [protected.id]
    assert [item.id for item in manager.get_context("session-a")] == [protected.id]
    assert window.total_tokens <= window.calculate_input_tokens()


@pytest.mark.parametrize("mandatory_kind", ["critical", "preserve_type"])
def test_selective_compressor_fails_closed_when_mandatory_content_exceeds_target(
    mandatory_kind,
):
    segment = _segment(
        "mandatory",
        "mandatory",
        100,
        session_id="session-a",
        priority=(
            ContextPriority.CRITICAL
            if mandatory_kind == "critical"
            else ContextPriority.LOW
        ),
    )
    kwargs = (
        {"preserve_types": [ContextType.CONVERSATION]}
        if mandatory_kind == "preserve_type"
        else {}
    )

    with pytest.raises(ValueError, match="Mandatory context .* exceeds target"):
        SelectiveCompressor().compress([segment], target_tokens=20, **kwargs)


class _RedisHashFake:
    def __init__(self, values):
        self.values = values

    def hgetall(self, key):
        return self.values


def test_redis_search_enforces_context_type_and_priority_filters():
    store = RedisContextStore(name="qualification-redis")
    matching = _segment(
        "matching",
        "shared evidence",
        10,
        session_id="session-a",
        priority=ContextPriority.HIGH,
    )
    wrong_type = matching.model_copy(deep=True)
    wrong_type.id = "wrong-type"
    wrong_type.type = ContextType.BACKGROUND
    wrong_priority = matching.model_copy(deep=True)
    wrong_priority.id = "wrong-priority"
    wrong_priority.priority = ContextPriority.LOW
    store._redis = _RedisHashFake(
        {
            item.id.encode(): store._serialize_segment(item)
            for item in (matching, wrong_type, wrong_priority)
        }
    )

    result = store.search(
        "shared evidence",
        "session-a",
        context_type=ContextType.CONVERSATION,
        priority=ContextPriority.HIGH,
    )

    assert [item.id for item in result] == ["matching"]


@pytest.mark.parametrize("backend", ["ram", "redis", "chroma"])
def test_store_add_rejects_explicit_session_lineage_mismatch_before_any_write(backend):
    valid = _segment("valid", "valid", 2, session_id="session-a")
    mismatch = _segment("mismatch", "mismatch", 2, session_id="session-b")

    if backend == "ram":
        store = RamContextStore(name="qualification-ram")
        write_target = None
    elif backend == "redis":
        store = RedisContextStore(name="qualification-redis")
        store._redis = MagicMock()
        write_target = store._redis
    else:
        store = ChromaContextStore(name="qualification-chroma")
        store._collection = MagicMock()
        write_target = store._collection

    with pytest.raises(ValueError, match="does not match segment session_id"):
        store.add([valid, mismatch], session_id="session-a")

    if backend == "ram":
        assert store._storage == {}
    else:
        write_target.assert_not_called()
        if backend == "redis":
            write_target.hset.assert_not_called()
        else:
            write_target.add.assert_not_called()
