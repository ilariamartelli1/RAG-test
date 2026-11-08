from collections.abc import Callable, Iterable

if "." in __package__:
    from ..models.chunk_models import ChunkPayload
    from ..models.record_models import SourceRecord
else:
    from models.chunk_models import ChunkPayload
    from models.record_models import SourceRecord


def ingest_records(
    loader: Callable[[], Iterable[SourceRecord]],
    chunker: Callable[[SourceRecord], Iterable[ChunkPayload]],
    store: Callable[[list[ChunkPayload]], int],
    chunk_batch_size: int,
) -> int:
    """
    Load records, chunk them, and store the chunks in bounded batches.

    Args:
        loader (Callable[[], Iterable[SourceRecord]]): Loads source records.
        chunker (Callable[[SourceRecord], Iterable[ChunkPayload]]): Splits a record.
        store (Callable[[list[ChunkPayload]], int]): Stores a batch and returns
            the number of new chunks stored.
        chunk_batch_size (int): Maximum number of chunks per storage call.

    Returns:
        int: Total number of new chunks stored.

    Raises:
        ValueError: If chunk_batch_size is not positive.

    Loader, chunker, and storage errors propagate; earlier batches may
    already have been stored when an error occurs.
    """
    if chunk_batch_size <= 0:
        raise ValueError("chunk_batch_size must be greater than zero")

    batch: list[ChunkPayload] = []
    stored = 0
    for record in loader():
        for chunk in chunker(record):
            batch.append(chunk)
            if len(batch) == chunk_batch_size:
                stored += store(batch)
                batch = []

    if batch:
        stored += store(batch)
    return stored
