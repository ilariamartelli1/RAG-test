"""Define source records shared by loaders and chunkers."""

from pydantic import BaseModel, JsonValue


class SourceRecord(BaseModel):
    """Content loaded from a source, before chunking.

    Content is either document text or a mapping of dataset field names to
    their original values. Metadata describes the source, such as a filename,
    page number, or dataset row identifier.
    """

    content: str | dict[str, object]
    metadata: dict[str, JsonValue]
