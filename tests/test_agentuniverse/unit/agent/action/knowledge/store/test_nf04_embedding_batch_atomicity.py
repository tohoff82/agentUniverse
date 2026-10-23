"""NF-04 / C-04A regression perimeter for embedding batch materialization.

These tests intentionally describe the required behavior before the repair exists:
an invalid generated-embedding cardinality must fail without mutating source
Documents or crossing the backend boundary. A retry after such a failure must
observe the same missing-document set.

The perimeter is independent of any upstream implementation patch.
"""

import asyncio
from unittest.mock import AsyncMock, Mock, call, patch

import pytest

from agentuniverse.agent.action.knowledge.store.document import Document
from agentuniverse.agent.action.knowledge.store.pgvector_store import PGVectorStore
from agentuniverse.agent.action.knowledge.store.redis_vector_store import RedisVectorStore


STORE_CASES = (
    (
        PGVectorStore,
        "agentuniverse.agent.action.knowledge.store.pgvector_store.EmbeddingManager",
        "_ensure_table",
        "_async_ensure_table",
        "pgvector",
    ),
    (
        RedisVectorStore,
        "agentuniverse.agent.action.knowledge.store.redis_vector_store.EmbeddingManager",
        "_ensure_index",
        "_async_ensure_index",
        "redis",
    ),
)

INVALID_BATCHES = (
    pytest.param([], id="zero"),
    pytest.param([[1.0, 0.0]], id="short"),
    pytest.param(
        [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
        id="oversized",
    ),
)


def _documents() -> list[Document]:
    return [
        Document(id="missing-a", text="missing-a"),
        Document(id="preembedded", text="preembedded", embedding=[9.0, 9.0]),
        Document(id="missing-b", text="missing-b"),
    ]


def _snapshot(documents: list[Document]) -> list[dict]:
    return [document.model_dump() for document in documents]


@pytest.mark.parametrize(
    "store_cls,manager_target,sync_boundary,async_boundary,store_id",
    STORE_CASES,
    ids=[case[-1] for case in STORE_CASES],
)
@pytest.mark.parametrize("generated", INVALID_BATCHES)
def test_invalid_embedding_count_is_atomic_sync(
    store_cls,
    manager_target,
    sync_boundary,
    async_boundary,
    store_id,
    generated,
):
    """Sync upsert must reject count mismatch before source or backend mutation."""
    del async_boundary, store_id
    documents = _documents()
    before = _snapshot(documents)
    model = Mock()
    model.get_embeddings.return_value = generated
    store = store_cls(embedding_model="f0-batch", dimensions=2)

    with patch(manager_target) as manager, patch.object(store, sync_boundary) as backend:
        manager.return_value.get_instance_obj.return_value = model

        with pytest.raises(ValueError):
            store.upsert_document(documents)

    assert _snapshot(documents) == before
    backend.assert_not_called()
    model.get_embeddings.assert_called_once_with(["missing-a", "missing-b"])


@pytest.mark.parametrize(
    "store_cls,manager_target,sync_boundary,async_boundary,store_id",
    STORE_CASES,
    ids=[case[-1] for case in STORE_CASES],
)
@pytest.mark.parametrize("generated", INVALID_BATCHES)
def test_invalid_embedding_count_is_atomic_async(
    store_cls,
    manager_target,
    sync_boundary,
    async_boundary,
    store_id,
    generated,
):
    """Async upsert must reject count mismatch before source or backend mutation."""
    del sync_boundary, store_id
    documents = _documents()
    before = _snapshot(documents)
    model = Mock()
    model.get_embeddings.return_value = generated
    store = store_cls(embedding_model="f0-batch", dimensions=2)

    with patch(manager_target) as manager, patch.object(
        store,
        async_boundary,
        new_callable=AsyncMock,
    ) as backend:
        manager.return_value.get_instance_obj.return_value = model

        with pytest.raises(ValueError):
            asyncio.run(store.async_upsert_document(documents))

    assert _snapshot(documents) == before
    backend.assert_not_awaited()
    model.get_embeddings.assert_called_once_with(["missing-a", "missing-b"])


@pytest.mark.parametrize(
    "store_cls,manager_target,sync_boundary,async_boundary,store_id",
    STORE_CASES,
    ids=[case[-1] for case in STORE_CASES],
)
def test_matching_embedding_batch_preserves_order_and_existing_vectors(
    store_cls,
    manager_target,
    sync_boundary,
    async_boundary,
    store_id,
):
    """A valid generated batch fills only missing positions in source order."""
    del sync_boundary, async_boundary, store_id
    documents = _documents()
    model = Mock()
    model.get_embeddings.return_value = [[1.0, 0.0], [0.0, 1.0]]
    store = store_cls(embedding_model="f0-batch", dimensions=2)

    with patch(manager_target) as manager:
        manager.return_value.get_instance_obj.return_value = model
        vectors = store._vectors_for_documents(documents)

    assert vectors == [[1.0, 0.0], [9.0, 9.0], [0.0, 1.0]]
    assert documents[1].embedding == [9.0, 9.0]
    model.get_embeddings.assert_called_once_with(["missing-a", "missing-b"])


@pytest.mark.parametrize(
    "store_cls,manager_target,sync_boundary,async_boundary,store_id",
    STORE_CASES,
    ids=[case[-1] for case in STORE_CASES],
)
def test_failed_embedding_batch_is_retry_stable(
    store_cls,
    manager_target,
    sync_boundary,
    async_boundary,
    store_id,
):
    """Repeated failed attempts must observe the same missing-document set."""
    del sync_boundary, async_boundary, store_id
    documents = _documents()
    before = _snapshot(documents)
    model = Mock()
    model.get_embeddings.return_value = [[1.0, 0.0]]
    store = store_cls(embedding_model="f0-batch", dimensions=2)

    with patch(manager_target) as manager:
        manager.return_value.get_instance_obj.return_value = model

        for _ in range(2):
            with pytest.raises(ValueError):
                store._vectors_for_documents(documents)

    assert _snapshot(documents) == before
    assert model.get_embeddings.call_args_list == [
        call(["missing-a", "missing-b"]),
        call(["missing-a", "missing-b"]),
    ]
