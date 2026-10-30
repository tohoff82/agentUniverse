import asyncio

from agentuniverse.agent.context.context_manager import ContextManager
from agentuniverse.agent.context.context_model import ContextType, ContextPriority
from agentuniverse.agent.context.store.ram_context_store import RamContextStore
from agentuniverse.agent.memory.memory import Memory
from agentuniverse.agent.memory.message import Message
from agentuniverse.agent.memory.memory_storage.ram_memory_storage import RamMemoryStorage
from agentuniverse.agent.memory.memory_storage.chroma_memory_storage import ChromaMemoryStorage
from agentuniverse.agent.memory.conversation_memory.conversation_message import ConversationMessage
from agentuniverse.agent.memory.conversation_memory.memory_storage.sqlite_conversation_memory_storage import SqliteMemoryStorage, DefaultMemoryConverter
from agentuniverse.agent.input_object import InputObject
from agentuniverse.agent.output_object import OutputObject
from agentuniverse.agent.template.implementation_agent_template import ImplementationAgentTemplate
from agentuniverse.agent.template.supervision_agent_template import SupervisionAgentTemplate
from agentuniverse.agent.work_pattern.is_work_pattern import ISWorkPattern


class NoOpCompressor:
    def compress(self, segments, target_tokens, **kwargs):
        return list(segments), None


class CapturingCollection:
    def __init__(self):
        self.add_calls = []

    def add(self, **kwargs):
        call = dict(kwargs)
        call["metadatas"] = [dict(item) for item in kwargs.get("metadatas") or []]
        self.add_calls.append(call)


class EmptyCorrectionImplementation(ImplementationAgentTemplate):
    def run(self, **kwargs):
        if kwargs.get("correction_mode"):
            return OutputObject({})
        return OutputObject({"output": "BASE", "checkpoint_output": "BASE"})

    async def async_run(self, **kwargs):
        return self.run(**kwargs)


class AlwaysCorrectionSupervision(SupervisionAgentTemplate):
    def run(self, **kwargs):
        return OutputObject({"needs_correction": True, "feedback": "fix", "score": 0.0})

    async def async_run(self, **kwargs):
        return self.run(**kwargs)


def _manager(max_tokens=1000, reserved_tokens=200, max_segments=100):
    m = ContextManager(name="hpf-context", default_max_tokens=max_tokens, default_reserved_tokens=reserved_tokens, enable_compression=True)
    m._hot_store = RamContextStore(name="hpf-ram", max_segments=max_segments)
    return m


def test_control_ram_context_session_isolation():
    s = RamContextStore(name="ctrl")
    from agentuniverse.agent.context.context_model import ContextSegment
    a = ContextSegment(type=ContextType.BACKGROUND, content="A", tokens=1, session_id="A")
    b = ContextSegment(type=ContextType.BACKGROUND, content="B", tokens=1, session_id="B")
    s.add([a], session_id="A")
    s.add([b], session_id="B")
    assert [x.content for x in s.get("A")] == ["A"]
    assert [x.content for x in s.get("B")] == ["B"]


def test_control_ram_memory_session_isolation():
    RamMemoryStorage.messages = {}
    s = RamMemoryStorage()
    s.add([Message(id="a", content="A")], session_id="A", agent_id="agent")
    s.add([Message(id="b", content="B")], session_id="B", agent_id="agent")
    assert [x.content for x in s.get(session_id="A", agent_id="agent")] == ["A"]
    assert [x.content for x in s.get(session_id="B", agent_id="agent")] == ["B"]


def test_control_supervision_positive_marker():
    assert SupervisionAgentTemplate()._extract_needs_correction("Correction required") is True


def test_context_budget_postcondition_after_noop_compression():
    m = _manager()
    w = m.create_context_window("s", max_tokens=1000, reserved_tokens=200)
    m.add_context("s", "a" * 2800, ContextType.BACKGROUND, ContextPriority.MEDIUM)
    m._compressor = NoOpCompressor()
    m.add_context("s", "b" * 800, ContextType.BACKGROUND, ContextPriority.MEDIUM)
    assert not w.is_over_budget(), (w.total_tokens, w.calculate_input_tokens())


def test_context_rejects_or_contains_oversized_single_segment():
    m = _manager(max_tokens=100, reserved_tokens=20)
    w = m.create_context_window("s", max_tokens=100, reserved_tokens=20)
    m.add_context("s", "x" * 400, ContextType.BACKGROUND)
    assert not w.is_over_budget(), (w.total_tokens, w.calculate_input_tokens())


def test_context_segment_inherits_window_scope():
    m = _manager()
    m.create_context_window("s", agent_id="agent-A", task_id="task-A")
    seg = m.add_context("s", "hello", ContextType.TASK)
    assert seg.agent_id == "agent-A"
    assert seg.task_id == "task-A"


def test_context_window_ledger_matches_store_after_lru_eviction():
    m = _manager(max_segments=1)
    w = m.create_context_window("s")
    m.add_context("s", "a" * 40, ContextType.BACKGROUND)
    m.add_context("s", "b" * 80, ContextType.BACKGROUND)
    stored = m._hot_store.get("s")
    assert set(w.segment_ids) == {x.id for x in stored}
    assert w.total_tokens == sum(x.tokens for x in stored)


def test_sqlite_delete_respects_session_agent_trace_scope():
    s = SqliteMemoryStorage(sqldb_path="sqlite:///:memory:", sqldb_table_name="memory")
    s.memory_converter = DefaultMemoryConverter("memory")
    s._new_client()

    def mk(i, sess, source, trace):
        return ConversationMessage(id=i, conversation_id=sess, source=source, source_type="agent", target="user", target_type="user", type="output", trace_id=trace, content=i, metadata={})

    s.add([mk("a", "s1", "agentA", "t1")], session_id="s1")
    s.add([mk("b", "s1", "agentB", "t2")], session_id="s1")
    s.add([mk("c", "s2", "agentB", "t3")], session_id="s2")
    s.delete(session_id="s1", agent_id="agentA", trace_id="t1")
    assert {m.id for m in s.get(top_k=20)} == {"b", "c"}


def test_explicit_empty_memory_storages_is_preserved():
    C = type("C", (), dict(name="m", description=None, type=None, memory_key=None, max_tokens=None, memory_compressor=None, memory_storages=[], memory_retrieval_storage="read", memory_summarize_agent=None, context_manager=None))
    m = Memory()
    m.initialize_by_component_configer(C())
    assert m.memory_storages == []


def test_chroma_batch_metadata_does_not_leak_source_to_sibling():
    c = CapturingCollection()
    s = ChromaMemoryStorage()
    s._collection = c
    s.embedding_model = None
    s.add([Message(id="a", content="A", source="agentA"), Message(id="b", content="B", source=None)], session_id="s", agent_id="agent")
    assert c.add_calls[0]["metadatas"][0].get("source") == "agentA"
    assert "source" not in c.add_calls[1]["metadatas"][0]


def test_supervision_explicit_false_marker_is_false():
    assert SupervisionAgentTemplate()._extract_needs_correction("Needs Correction: false") is False


def test_empty_correction_is_not_counted_as_applied_sync():
    p = ISWorkPattern()
    p.implementation = EmptyCorrectionImplementation()
    p.supervision = AlwaysCorrectionSupervision()
    result = p.invoke(InputObject({"input": "x"}), {"input": "x", "checkpoint_count": 1, "max_corrections": 1})
    assert result["execution_context"]["corrections_made"] == 0
    assert result["execution_context"]["checkpoint_history"][0]["corrected"] is False


def test_empty_correction_is_not_counted_as_applied_async():
    p = ISWorkPattern()
    p.implementation = EmptyCorrectionImplementation()
    p.supervision = AlwaysCorrectionSupervision()
    result = asyncio.run(p.async_invoke(InputObject({"input": "x"}), {"input": "x", "checkpoint_count": 1, "max_corrections": 1}))
    assert result["execution_context"]["corrections_made"] == 0
    assert result["execution_context"]["checkpoint_history"][0]["corrected"] is False
