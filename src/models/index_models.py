"""Define document index settings and build identity."""

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.utils.id_utils import fingerprint_document


def _require_nonblank(value: str) -> str:
    if not value.strip():
        raise ValueError("must not be empty or whitespace")
    return value


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
        return _require_nonblank(value)


class IndexBuildManifest(BaseModel):
    """Identify a document index build and the settings it uses.

    Identifiers are supplied by the caller. This model does not establish
    ownership, ensure uniqueness, track progress, or persist the manifest.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: str
    build_id: str
    collection_name: str
    index_settings: IndexSettings

    @field_validator("document_id", "build_id", "collection_name")
    @classmethod
    def require_nonblank(cls, value: str) -> str:
        return _require_nonblank(value)


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
