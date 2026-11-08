from .record_models import SourceRecord


def load_text_records(data: bytes, source: str) -> list[SourceRecord]:
    """
    Decode a UTF-8 text document as a single source record.

    Args:
        data (bytes): Contents of the text document.
        source (str): Source identifier, such as the uploaded filename.

    Returns:
        list[SourceRecord]: One record containing the document text.

    Raises:
        ValueError: If source is empty or contains only whitespace.
        UnicodeDecodeError: If data is not valid UTF-8.
    """
    if not source.strip():
        raise ValueError("Text document source must not be empty")

    return [SourceRecord(content=data.decode("utf-8-sig"), metadata={"source": source})]
