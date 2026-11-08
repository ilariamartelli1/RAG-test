"""Extract text-based PDF pages into source records without OCR."""

from io import BytesIO

from pypdf import PdfReader

if "." in __package__:
    from ..models.record_models import SourceRecord
else:
    from models.record_models import SourceRecord


def load_pdf_records(data: bytes, source: str) -> list[SourceRecord]:
    """
    Extract text from PDF pages as source records.

    Args:
        data (bytes): Contents of the PDF document.
        source (str): Source identifier, such as the uploaded filename.

    Returns:
        list[SourceRecord]: One record per page, with one-based page numbers.

    Raises:
        ValueError: If source is empty, the PDF is encrypted, or the document
            contains no extractable text. PDF parsing errors propagate.

    Empty pages are retained. Image-only pages require OCR, which is not
    supported by this loader.
    """
    if not source.strip():
        raise ValueError("PDF document source must not be empty")

    reader = PdfReader(BytesIO(data))
    if reader.is_encrypted:
        raise ValueError("Encrypted PDF documents are not supported")

    records = [
        SourceRecord(
            content=page.extract_text(),
            metadata={"source": source, "page": page_number},
        )
        for page_number, page in enumerate(reader.pages, start=1)
    ]
    if not any(isinstance(record.content, str) and record.content.strip() for record in records):
        raise ValueError("PDF document contains no extractable text; OCR is not supported")
    return records
