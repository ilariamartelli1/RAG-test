from dataclasses import dataclass

import pandas as pd

if __package__:
    from .text_utils import safe_str
else:
    from text_utils import safe_str


@dataclass(frozen=True)
class DatasetChunk:
    text: str
    row_id: str
    chunk_index: int
    field_name: str | None


def _value_to_text(value: object) -> str:
    return safe_str(value) if pd.api.types.is_scalar(value) else str(value)


def row_to_text(row: pd.Series) -> str:
    """
    Convert a dataset row to a formatted text string.
    Args:
        row (pd.Series): Dataset row.
    Returns:
        str: Formatted row text.
    """
    fields = (
        f"{name}: {_value_to_text(value)}"
        for name, value in row.items()
    )
    return "\n".join(fields).strip()


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
        field_text = _value_to_text(value)
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
