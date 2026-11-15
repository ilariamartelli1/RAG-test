"""Build a new document index without reusing existing collections."""

from functools import partial

from qdrant_client import QdrantClient

from .document_ingestion import ingest_document

from src.embeddings.embedding_provider import EmbeddingProvider, get_embedding_dimension
from src.models.chunk_models import ChunkPayload
from src.models.index_models import build_index_settings
from src.storage.chunk_service import store_chunks
from src.storage.collection_service import create_collection_with_settings, mark_collection_index_ready


def build_document_index(
    *,
    qdrant: QdrantClient,
    collection_name: str,
    data: bytes,
    filename: str,
    embedder: EmbeddingProvider,
    embedding_provider: str,
    embedding_model: str,
    embedding_batch_size: int,
    chunk_batch_size: int,
    max_chunk_chars: int = 1000,
) -> int:
    """Create and populate a new collection, marking it ready after all writes.

    The caller must exclusively own the indexing operation and supply the
    identity of the actual embedder. Failures after creation leave the
    collection in place, potentially with partial data. If marking ready
    fails, its status may be unknown and must be checked before use.
    """
    if chunk_batch_size <= 0:
        raise ValueError("chunk_batch_size must be greater than zero")
    settings = build_index_settings(
        document_bytes=data,
        document_filename=filename,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        max_chunk_chars=max_chunk_chars,
    )
    dimension = get_embedding_dimension(embedder, embedding_batch_size)
    create_collection_with_settings(qdrant, collection_name, dimension, settings)

    store = partial(
        store_chunks, qdrant, embedder, collection_name,
        embedding_batch_size=embedding_batch_size,
    )
    has_text = False

    def store_batch(chunks: list[ChunkPayload]) -> int:
        nonlocal has_text
        has_text = has_text or any(chunk.text.strip() for chunk in chunks)
        return store(chunks)

    stored = ingest_document(
        data, filename, store_batch, chunk_batch_size, max_chunk_chars
    )
    if not has_text:
        raise ValueError("Document contains no extractable text")
    mark_collection_index_ready(qdrant, collection_name, settings)
    return stored
