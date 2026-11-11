"""Define the settings used to build a document index."""

from pydantic import BaseModel, ConfigDict, Field, field_validator

if "." in __package__:
    from ..utils.id_utils import fingerprint_document
else:
    from utils.id_utils import fingerprint_document


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


def build_index_settings(
    *,
    document_bytes: bytes,
    document_filename: str,
    embedding_provider: str,
    embedding_model: str,
    max_chunk_chars: int,
) -> IndexSettings:
    """Create validated index settings from the original document bytes."""
    return IndexSettings(
        document_fingerprint=fingerprint_document(document_bytes),
        document_filename=document_filename,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        max_chunk_chars=max_chunk_chars,
    )


def compare_index_settings(saved: IndexSettings, current: IndexSettings) -> list[str]:
    """Return differing field names in model order, without changing either input.

    Matching settings do not establish that indexing completed or that model
    weights are unchanged.
    """
    saved_fields = saved.model_dump()
    current_fields = current.model_dump()
    return [field for field in saved_fields if saved_fields[field] != current_fields[field]]
