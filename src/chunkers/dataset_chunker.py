from dataclasses import dataclass

import pandas as pd

if "." in __package__:
    from ..chunk_models import ChunkPayload
    from ..processing.dataset_utils import row_to_text, value_to_text
    from ..processing.record_models import SourceRecord
else:
    from chunk_models import ChunkPayload
    from processing.dataset_utils import row_to_text, value_to_text
    from processing.record_models import SourceRecord


@dataclass(frozen=True)
class DatasetChunk:
    text: str
    row_id: str
    chunk_index: int
    field_name: str | None


def chunk_dataset_row(
    row: pd.Series, row_id: str, max_chunk_chars: int = 1000
) -> list[DatasetChunk]:
    """
    Keep a row together if it fits; otherwise split within individual fields.

    Args:
        row (pd.Series): Dataset row to split.
        row_id (str): Identifier shared by all chunks from the row.
        max_chunk_chars (int): Maximum characters per chunk, including labels.

    Returns:
        list[DatasetChunk]: Chunks containing text and row metadata.
    """
    if max_chunk_chars <= 0:
        raise ValueError("max_chunk_chars must be greater than zero")
    if not row_id:
        raise ValueError("row_id must not be empty")
    if row.empty:
        return []

    header = f"Record: {row_id}\n"
    text = header + row_to_text(row)
    if len(text) <= max_chunk_chars:
        return [DatasetChunk(text, row_id, 0, None)]

    chunks: list[DatasetChunk] = []
    for name, value in row.items():
        field_name = str(name)
        prefix = f"{header}{field_name}: "
        field_text = value_to_text(value)
        available = max_chunk_chars - len(prefix)
        if available < 0 or (available == 0 and field_text):
            raise ValueError("max_chunk_chars is too small for the record and field labels")

        if not field_text:
            chunks.append(DatasetChunk(prefix, row_id, len(chunks), field_name))
            continue

        while field_text:
            end = min(available, len(field_text))
            if end < len(field_text):
                for position in range(end - 1, -1, -1):
                    if field_text[position].isspace():
                        end = position + 1
                        break
            chunks.append(
                DatasetChunk(prefix + field_text[:end], row_id, len(chunks), field_name)
            )
            field_text = field_text[end:]

    return chunks


def chunk_dataset_record(
    record: SourceRecord, max_chunk_chars: int = 1000
) -> list[ChunkPayload]:
    """
    Split a dataset record into chunks while preserving its source metadata.

    Args:
        record (SourceRecord): Dataset fields and metadata, including a string row_id.
        max_chunk_chars (int): Maximum characters per chunk, including labels.

    Returns:
        list[ChunkPayload]: Text chunks with source and chunk metadata.

    Raises:
        ValueError: If the record is not a dataset row, lacks a valid row_id,
            or contains reserved chunk metadata.
    """
    if not isinstance(record.content, dict):
        raise ValueError("Dataset record content must be a field mapping")
    row_id = record.metadata.get("row_id")
    if not isinstance(row_id, str) or not row_id:
        raise ValueError("Dataset record metadata must contain a non-empty string row_id")
    if "chunk_index" in record.metadata or "field_name" in record.metadata:
        raise ValueError("Dataset record metadata contains reserved chunk fields")

    row = pd.Series(record.content, dtype=object)
    return [
        ChunkPayload(
            text=chunk.text,
            metadata={
                **record.metadata,
                "chunk_index": chunk.chunk_index,
                "field_name": chunk.field_name,
            },
        )
        for chunk in chunk_dataset_row(row, row_id, max_chunk_chars)
    ]
