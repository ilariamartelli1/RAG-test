from collections.abc import Callable
from functools import partial

from .ingestion_service import ingest_records

if "." in __package__:
    from ..models.chunk_models import ChunkPayload
    from ..chunkers.text_chunker import chunk_text_record
    from ..loaders.document_loader import load_document_records
else:
    from models.chunk_models import ChunkPayload
    from chunkers.text_chunker import chunk_text_record
    from loaders.document_loader import load_document_records


def ingest_document(
    data: bytes,
    filename: str,
    store: Callable[[list[ChunkPayload]], int],
    chunk_batch_size: int,
    max_chunk_chars: int = 1000,
) -> int:
    """
    Load, chunk and store a document using the configured batch sizes.

    Args:
        data (bytes): Contents of the document.
        filename (str): Filename used to select the loader and identify the source.
        store (Callable[[list[ChunkPayload]], int]): Stores a batch and returns
            the number of new chunks stored.
        chunk_batch_size (int): Maximum number of chunks per storage call.
        max_chunk_chars (int): Maximum number of characters per chunk.

    Returns:
        int: Total number of new chunks stored.

    Raises:
        ValueError: If either size is not positive.

    Loading, chunking and storage errors propagate; earlier batches may
    already have been stored when an error occurs.
    """
    if max_chunk_chars <= 0:
        raise ValueError("max_chunk_chars must be greater than zero")

    return ingest_records(
        loader=partial(load_document_records, data, filename),
        chunker=partial(chunk_text_record, max_chunk_chars=max_chunk_chars),
        store=store,
        chunk_batch_size=chunk_batch_size,
    )
