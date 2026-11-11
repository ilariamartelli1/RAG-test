"""Define the settings used to build a document index."""

from pydantic import BaseModel, ConfigDict, Field, field_validator


class IndexSettings(BaseModel):
    """Describe the document and embedding settings used for an index.

    The fingerprint is the lowercase SHA-256 digest of the original document
    bytes. The filename affects loader selection and stored source metadata.
    Model names do not guarantee that a model's underlying weights are
    unchanged.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    document_fingerprint: str = Field(
        min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$"
    )
    document_filename: str
    embedding_provider: str
    embedding_model: str
    max_chunk_chars: int = Field(gt=0, strict=True)

    @field_validator("document_filename", "embedding_provider", "embedding_model")
    @classmethod
    def require_nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty or whitespace")
        return value
