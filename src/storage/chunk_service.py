from __future__ import annotations

import json
from typing import TYPE_CHECKING

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

if "." in __package__:
    from ..id_utils import make_id
    from ..chunk_models import ChunkPayload
else:
    from id_utils import make_id
    from chunk_models import ChunkPayload

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


def store_chunks(
    qdrant: QdrantClient,
    embedder: SentenceTransformer,
    collection: str,
    chunks: list[ChunkPayload],
    embedding_batch_size: int,
) -> int:
    """
    Embed and store chunks that are not already in the collection.

    Args:
        qdrant (QdrantClient): Client connected to Qdrant.
        embedder (SentenceTransformer): Model used to embed chunk text.
        collection (str): Name of the target Qdrant collection.
        chunks (list[ChunkPayload]): Text and metadata to store.
        embedding_batch_size (int): Batch size used by the embedding model.

    Returns:
        int: Number of new chunks stored.
    """
    if not chunks:
        return 0

    chunks_by_id = {
        make_id(json.dumps(chunk.model_dump(), sort_keys=True, ensure_ascii=False)): chunk
        for chunk in chunks
    }
    existing = qdrant.retrieve(collection_name=collection, ids=list(chunks_by_id))
    existing_ids = {point.id for point in existing}
    new_chunks = [
        (point_id, chunk)
        for point_id, chunk in chunks_by_id.items()
        if point_id not in existing_ids
    ]
    if not new_chunks:
        return 0

    embeddings = embedder.encode(
        [chunk.text for _, chunk in new_chunks],
        batch_size=embedding_batch_size,
        show_progress_bar=False,
    ).tolist()
    points = [
        PointStruct(id=point_id, vector=vector, payload=chunk.model_dump())
        for (point_id, chunk), vector in zip(new_chunks, embeddings, strict=True)
    ]
    qdrant.upsert(collection_name=collection, points=points)
    return len(points)


def retrieve_chunks(
    qdrant: QdrantClient,
    embedder: SentenceTransformer,
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
        embedder (SentenceTransformer): Model used to embed the query.
        collection (str): Name of the Qdrant collection to search.
        query (str): Question or text to search for.
        embedding_batch_size (int): Batch size used by the embedding model.
        top_n (int): Maximum number of chunks to retrieve.
        score_threshold (float): Minimum similarity score for returned chunks.

    Returns:
        list[tuple[ChunkPayload, float]]: Chunks and their similarity scores.
    """
    query_vector = embedder.encode(
        query,
        batch_size=embedding_batch_size,
        show_progress_bar=False,
    ).tolist()
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
