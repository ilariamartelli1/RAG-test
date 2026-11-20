"""Build a new document index without reusing existing collections."""

from functools import partial

from qdrant_client import QdrantClient

from .ingestion_service import ingest_records

from src.chunkers.text_chunker import chunk_text_record
from src.embeddings.embedding_provider import EmbeddingProvider, get_embedding_dimension
from src.loaders.document_loader import load_document_records
from src.models.index_models import IndexBuildManifest, build_index_settings
from src.storage.chunk_service import store_chunks
from src.storage.collection_service import create_collection_with_settings, mark_collection_index_ready


def build_document_index(
    *,
    qdrant: QdrantClient,
    collection_name: str,
    document_id: str,
    build_id: str,
    data: bytes,
    filename: str,
    embedder: EmbeddingProvider,
    embedding_provider: str,
    embedding_model: str,
    embedding_batch_size: int,
    chunk_batch_size: int,
    max_chunk_chars: int = 1000,
) -> int:
    """Validate a document, then embed and store it in a new collection.

    The caller must exclusively own the indexing operation and supply the
    identity of the actual embedder and a unique build ID. IDs are recorded
    but their uniqueness and ownership are not verified. Document records are
    loaded in memory; embeddings are stored in batches. Failures after creation
    leave the collection in place, potentially with partial data. If marking
    ready fails, its status may be unknown. Resuming a failed build is not
    supported.
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
    manifest = IndexBuildManifest(
        document_id=document_id,
        build_id=build_id,
        collection_name=collection_name,
        index_settings=settings,
    )
    records = load_document_records(data, filename)
    if not any(
        isinstance(record.content, str) and record.content.strip()
        for record in records
    ):
        raise ValueError("Document contains no extractable text")

    dimension = get_embedding_dimension(embedder, embedding_batch_size)
    create_collection_with_settings(
        qdrant, collection_name, dimension, settings, manifest=manifest
    )
    stored = ingest_records(
        loader=lambda: records,
        chunker=partial(chunk_text_record, max_chunk_chars=max_chunk_chars),
        store=partial(
            store_chunks, qdrant, embedder, collection_name,
            embedding_batch_size=embedding_batch_size, embedding_dimension=dimension,
        ),
        chunk_batch_size=chunk_batch_size,
    )
    mark_collection_index_ready(qdrant, collection_name, settings)
    return stored
