"""Define the chunk payload shared by ingestion, storage and retrieval."""

from pydantic import BaseModel, JsonValue


class ChunkPayload(BaseModel):
    """Text and source metadata stored with a vector."""

    text: str
    metadata: dict[str, JsonValue]
