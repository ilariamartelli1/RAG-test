from pathlib import Path

from .pdf_loader import load_pdf_records
from .record_models import SourceRecord
from .text_loader import load_text_records


def load_document_records(data: bytes, filename: str) -> list[SourceRecord]:
    """
    Load document records using the filename extension.

    Args:
        data (bytes): Contents of the document.
        filename (str): Document filename, also used as the source identifier.

    Returns:
        list[SourceRecord]: Records produced by the appropriate loader.

    Raises:
        ValueError: If the filename is empty or its extension is unsupported.
            Loader errors propagate.
    """
    if not filename.strip():
        raise ValueError("Document filename must not be empty")

    extension = Path(filename).suffix.lower()
    if extension in (".txt", ".md"):
        return load_text_records(data, filename)
    if extension == ".pdf":
        return load_pdf_records(data, filename)
    raise ValueError(f"Unsupported document extension: {extension!r}")
