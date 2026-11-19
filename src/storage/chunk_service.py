"""Embed, store and retrieve chunk payloads using Qdrant."""

from __future__ import annotations

import json
import math
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, UpdateStatus

from src.utils.id_utils import make_id
from src.models.chunk_models import ChunkPayload
from src.embeddings.embedding_provider import EmbeddingProvider
from src.models.index_models import IndexSettings
from src.storage.collection_service import ensure_document_index_compatible


def chunks_by_id(chunks: list[ChunkPayload]) -> dict[int, ChunkPayload]:
    """Group chunks by their deterministic point ID, keeping the last duplicate."""
    return {
        make_id(json.dumps(chunk.model_dump(), sort_keys=True, ensure_ascii=False)): chunk
        for chunk in chunks
    }


def upsert_chunk_points(
    qdrant: QdrantClient, collection: str, points: list[PointStruct]
) -> None:
    """Write points and require Qdrant to confirm completion."""
    result = qdrant.upsert(collection_name=collection, points=points, wait=True)
    if result.status != UpdateStatus.COMPLETED:
        raise RuntimeError(
            f"Qdrant did not complete the chunk write for collection {collection!r}: "
            f"{result.status}"
        )


def _validate_embeddings(
    embeddings: list[list[float]], expected_count: int, embedding_dimension: int
) -> None:
    if len(embeddings) != expected_count:
        raise ValueError("Embedding provider must return one vector per chunk")
    if any(
        len(vector) != embedding_dimension
        or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            for value in vector
        )
        for vector in embeddings
    ):
        raise ValueError(
            "Embedding vectors must contain finite numbers and match the query dimension"
        )


def store_chunks(
    qdrant: QdrantClient,
    embedder: EmbeddingProvider,
    collection: str,
    chunks: list[ChunkPayload],
    embedding_batch_size: int,
    *,
    embedding_dimension: int | None = None,
) -> int:
    """
    Embed and store chunks that are not already in the collection.

    Args:
        qdrant (QdrantClient): Client connected to Qdrant.
        embedder (EmbeddingProvider): Provider used to embed chunk text.
        collection (str): Name of the target Qdrant collection.
        chunks (list[ChunkPayload]): Text and metadata to store.
        embedding_batch_size (int): Batch size used by the embedding model.
        embedding_dimension (int | None): Expected vector size, if known.

    Returns:
        int: Number of new chunks stored.
    """
    if embedding_dimension is not None and embedding_dimension <= 0:
        raise ValueError("embedding_dimension must be greater than zero")
    if not chunks:
        return 0

    indexed_chunks = chunks_by_id(chunks)
    existing = qdrant.retrieve(collection_name=collection, ids=list(indexed_chunks))
    existing_ids = {point.id for point in existing}
    new_chunks = [
        (point_id, chunk)
        for point_id, chunk in indexed_chunks.items()
        if point_id not in existing_ids
    ]
    if not new_chunks:
        return 0

    embeddings = embedder.embed_documents(
        [chunk.text for _, chunk in new_chunks], embedding_batch_size
    )
    if embedding_dimension is not None:
        _validate_embeddings(embeddings, len(new_chunks), embedding_dimension)
    points = [
        PointStruct(id=point_id, vector=vector, payload=chunk.model_dump())
        for (point_id, chunk), vector in zip(new_chunks, embeddings, strict=True)
    ]
    upsert_chunk_points(qdrant, collection, points)
    return len(points)


def retrieve_chunks(
    qdrant: QdrantClient,
    embedder: EmbeddingProvider,
    collection: str,
    query: str,
    embedding_batch_size: int,
    top_n: int = 5,
    score_threshold: float = 0.2,
) -> list[tuple[ChunkPayload, float]]:
    """
    Retrieve validated chunks and their similarity scores.

    Args:
        qdrant (QdrantClient): Client connected to Qdrant.
        embedder (EmbeddingProvider): Provider used to embed the query.
        collection (str): Name of the Qdrant collection to search.
        query (str): Question or text to search for.
        embedding_batch_size (int): Batch size used by the embedding model.
        top_n (int): Maximum number of chunks to retrieve.
        score_threshold (float): Minimum similarity score for returned chunks.

    Returns:
        list[tuple[ChunkPayload, float]]: Chunks and their similarity scores.
    """
    query_vector = embedder.embed_query(query, embedding_batch_size)
    return _query_chunks(qdrant, collection, query_vector, top_n, score_threshold)


def retrieve_document_chunks(
    qdrant: QdrantClient,
    embedder: EmbeddingProvider,
    collection: str,
    settings: IndexSettings,
    query: str,
    embedding_batch_size: int,
    top_n: int = 5,
    score_threshold: float = 0.2,
) -> list[tuple[ChunkPayload, float]]:
    """Retrieve from a ready, compatible document collection.

    The caller must supply settings matching the actual embedder and prevent
    concurrent collection changes; the compatibility checks are not atomic.
    """
    if embedding_batch_size <= 0:
        raise ValueError("embedding_batch_size must be greater than zero")
    query_vector = embedder.embed_query(query, embedding_batch_size)
    if not query_vector:
        raise ValueError("Embedding provider returned an empty vector")
    ensure_document_index_compatible(qdrant, collection, len(query_vector), settings)
    return _query_chunks(qdrant, collection, query_vector, top_n, score_threshold)


def _query_chunks(
    qdrant: QdrantClient,
    collection: str,
    query_vector: list[float],
    top_n: int,
    score_threshold: float,
) -> list[tuple[ChunkPayload, float]]:
    results = qdrant.query_points(
        collection_name=collection,
        query=query_vector,
        limit=top_n,
        with_payload=True,
        score_threshold=score_threshold,
    )
    return [
        (ChunkPayload.model_validate(point.payload), point.score)
        for point in results.points
    ]
