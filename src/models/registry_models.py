"""Define records for a future document and build registry."""

from pydantic import BaseModel, ConfigDict, field_validator

from src.models.index_models import IndexBuildManifest


class DocumentRegistryRecord(BaseModel):
    """Identify a logical document independently of its individual builds."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: str

    @field_validator("document_id")
    @classmethod
    def require_nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty or whitespace")
        return value


class BuildRegistryRecord(BaseModel):
    """Describe a build using its existing identity and index settings.

    Neither registry record persists data or establishes uniqueness or ownership.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    manifest: IndexBuildManifest
