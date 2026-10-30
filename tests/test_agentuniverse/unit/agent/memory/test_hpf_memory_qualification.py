"""Hermetic persistent-memory qualification oracle (AU-YELLOW-CLOSURE-001).

The oracle deliberately exercises the shipped SQLite conversation-memory API.
It uses only a temporary local database and makes no network or LLM calls.
The ``HOLD`` cases document semantics that the implementation does not expose.
"""

from pathlib import Path

import pytest

from agentuniverse.agent.memory.conversation_memory.conversation_message import (
    ConversationMessage,
)
from agentuniverse.agent.memory.conversation_memory.memory_storage.sqlite_conversation_memory_storage import (
    DefaultMemoryConverter,
    SqliteMemoryStorage,
)


def _storage(database: Path) -> SqliteMemoryStorage:
    """Build the production storage directly, without component discovery."""
    storage = SqliteMemoryStorage.model_construct(
        sqldb_table_name="hpf_memory",
        sqldb_path=f"sqlite:///{database}",
        memory_converter=DefaultMemoryConverter("hpf_memory"),
        engine=None,
        session=None,
    )
    storage._new_client()
    return storage


def _message(
    message_id: str,
    content: str | None,
    *,
    agent: str,
    trace_id: str | None = None,
    additional_args: dict | None = None,
) -> ConversationMessage:
    return ConversationMessage(
        id=message_id,
        trace_id=trace_id or f"trace-{message_id}",
        source="user",
        source_type="user",
        target=agent,
        target_type="agent",
        type="input",
        content=content,
        metadata={"prefix": "", "params": "{}", "pair_id": message_id},
        additional_args={} if additional_args is None else additional_args,
    )


def _contents(messages: list[ConversationMessage]) -> list[str | None]:
    return [message.content for message in messages]


def test_mem_01_sessions_are_isolated(tmp_path: Path) -> None:
    storage = _storage(tmp_path / "mem01.sqlite")
    storage.add([_message("a", "only-a", agent="agent")], session_id="session-a")
    storage.add([_message("b", "only-b", agent="agent")], session_id="session-b")

    assert _contents(storage.get(session_id="session-a")) == ["only-a"]
    assert _contents(storage.get(session_id="session-b")) == ["only-b"]


def test_mem_02_agents_are_isolated_when_agent_scope_is_requested(tmp_path: Path) -> None:
    storage = _storage(tmp_path / "mem02.sqlite")
    storage.add([_message("a", "only-a", agent="agent-a")], session_id="session")
    storage.add([_message("b", "only-b", agent="agent-b")], session_id="session")

    assert _contents(storage.get(session_id="session", agent_id="agent-a")) == ["only-a"]
    assert _contents(storage.get(session_id="session", agent_id="agent-b")) == ["only-b"]


def test_mem_03_local_sqlite_restart_restores_fidelity(tmp_path: Path) -> None:
    database = tmp_path / "mem03.sqlite"
    first = _storage(database)
    original = _message(
        "durable",
        "survives restart",
        agent="agent-a",
        trace_id="trace-durable",
        additional_args={"nested": {"ok": True}, "step": 3},
    )
    first.add([original], session_id="session-a")
    first.engine.dispose()

    restored_storage = _storage(database)
    restored, = restored_storage.get(session_id="session-a", agent_id="agent-a")

    assert restored.id == original.id
    assert restored.content == original.content
    assert restored.trace_id == original.trace_id
    assert restored.source == original.source
    assert restored.source_type == original.source_type
    assert restored.target == original.target
    assert restored.target_type == original.target_type
    assert restored.type == original.type
    assert restored.additional_args == original.additional_args
    assert restored.conversation_id == "session-a"


def test_mem_04_absent_record_and_explicit_empty_content_remain_distinct(tmp_path: Path) -> None:
    storage = _storage(tmp_path / "mem04.sqlite")
    storage.add([_message("empty", "", agent="agent")], session_id="has-empty")

    assert storage.get(session_id="absent") == []
    restored, = storage.get(session_id="has-empty")
    assert restored.content == ""


def test_mem_05_delete_keeps_all_unrequested_session_and_agent_scopes(tmp_path: Path) -> None:
    storage = _storage(tmp_path / "mem05.sqlite")
    storage.add([_message("a-a", "delete", agent="agent-a")], session_id="session-a")
    storage.add([_message("a-b", "keep-agent", agent="agent-b")], session_id="session-a")
    storage.add([_message("b-a", "keep-session", agent="agent-a")], session_id="session-b")

    storage.delete(session_id="session-a", agent_id="agent-a")

    assert storage.get(session_id="session-a", agent_id="agent-a") == []
    assert _contents(storage.get(session_id="session-a", agent_id="agent-b")) == ["keep-agent"]
    assert _contents(storage.get(session_id="session-b", agent_id="agent-a")) == ["keep-session"]


def test_mem_05_trace_only_delete_is_scoped_and_no_scope_delete_is_a_noop(tmp_path: Path) -> None:
    storage = _storage(tmp_path / "mem05-trace.sqlite")
    storage.add(
        [_message("delete", "delete", agent="agent-a", trace_id="trace-delete")],
        session_id="session-a",
    )
    storage.add(
        [_message("keep", "keep", agent="agent-b", trace_id="trace-keep")],
        session_id="session-b",
    )

    storage.delete(trace_id="trace-delete")

    assert storage.get(trace_id="trace-delete") == []
    assert _contents(storage.get(trace_id="trace-keep")) == ["keep"]

    storage.delete()

    assert _contents(storage.get(trace_id="trace-keep")) == ["keep"]


@pytest.mark.skip(reason="MEM-05 HOLD: the memory storage API exposes no update operation")
def test_mem_05_update_scope_hold() -> None:
    pass


@pytest.mark.skip(reason="MEM-06 HOLD: no restoration version or stale-write ordering is represented")
def test_mem_06_stale_restoration_version_ordering_hold() -> None:
    pass


def test_mem_07_failed_batch_is_raised_and_not_applied(tmp_path: Path) -> None:
    storage = _storage(tmp_path / "mem07.sqlite")
    valid = _message("valid", "must roll back", agent="agent")
    invalid = _message(
        "invalid",
        "cannot serialize",
        agent="agent",
        additional_args={"not-json": {1, 2, 3}},
    )

    with pytest.raises(TypeError):
        storage.add([valid, invalid], session_id="session")

    assert storage.get(session_id="session") == []


def test_mem_08_no_match_and_zero_limit_never_broaden_retrieval(tmp_path: Path) -> None:
    storage = _storage(tmp_path / "mem08.sqlite")
    storage.add([_message("a", "only-a", agent="agent")], session_id="session-a")
    storage.add([_message("b", "only-b", agent="agent")], session_id="session-b")

    assert storage.get(session_id="does-not-exist") == []
    assert storage.get(session_id="session-a", top_k=0) == []
    assert storage.get(session_id="session-a", top_k=-1) == []
