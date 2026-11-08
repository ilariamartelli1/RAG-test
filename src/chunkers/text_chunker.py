if "." in __package__:
    from ..chunk_models import ChunkPayload
    from ..processing.record_models import SourceRecord
else:
    from chunk_models import ChunkPayload
    from processing.record_models import SourceRecord


def chunk_text_record(
    record: SourceRecord, max_chunk_chars: int = 1000
) -> list[ChunkPayload]:
    """
    Split a text record into bounded chunks, preserving text and metadata.

    Splits prefer whitespace boundaries when possible, and fall back to
    splitting within a word. Concatenating the chunks reproduces the text.

    Args:
        record (SourceRecord): Text content and source metadata.
        max_chunk_chars (int): Maximum number of characters per chunk.

    Returns:
        list[ChunkPayload]: Ordered chunks with zero-based chunk indices.

    Raises:
        ValueError: If the record is not text, its metadata contains
            chunk_index, or max_chunk_chars is not positive.
    """
    if not isinstance(record.content, str):
        raise ValueError("Text record content must be a string")
    if "chunk_index" in record.metadata:
        raise ValueError("Text record metadata contains reserved chunk_index")
    if max_chunk_chars <= 0:
        raise ValueError("max_chunk_chars must be greater than zero")

    text = record.content
    chunks: list[ChunkPayload] = []
    start = 0
    while start < len(text):
        end = min(start + max_chunk_chars, len(text))
        if end < len(text):
            for position in range(end - 1, start - 1, -1):
                if text[position].isspace():
                    end = position + 1
                    break
        chunks.append(ChunkPayload(
            text=text[start:end],
            metadata={**record.metadata, "chunk_index": len(chunks)},
        ))
        start = end
    return chunks
