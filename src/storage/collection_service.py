"""Create or check a Qdrant collection for chunk embeddings."""

from typing import Literal, get_args

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from pydantic import ValidationError

from src.models.index_models import IndexBuildManifest, IndexSettings, compare_index_settings


INDEX_SETTINGS_METADATA_KEY = "index_settings"
INDEX_STATUS_METADATA_KEY = "index_status"
INDEX_BUILD_MANIFEST_METADATA_KEY = "index_build_manifest"
IndexStatus = Literal["building", "ready"]


class IndexSettingsMismatchError(ValueError):
    """Report which stored index settings differ from the current settings."""

    def __init__(self, collection_name: str, differences: list[str]) -> None:
        self.collection_name = collection_name
        self.differences = tuple(differences)
        super().__init__(
            f"Collection {collection_name!r} has different index settings: "
            f"{', '.join(self.differences)}. Use matching settings or a different "
            "collection, or explicitly rebuild the index."
        )


class IndexSettingsMetadataError(ValueError):
    """Report missing or invalid collection index settings metadata."""

    def __init__(
        self, collection_name: str, reason: Literal["missing", "invalid"], message: str
    ) -> None:
        self.collection_name = collection_name
        self.reason = reason
        super().__init__(message)


def get_collection_build_manifest(
    qdrant: QdrantClient, collection_name: str
) -> IndexBuildManifest:
    """Read a build manifest and check it agrees with collection metadata.

    The collection name must be the physical name stored in the manifest, not
    an alias. Missing, invalid or inconsistent metadata raises ValueError.
    This does not establish ownership, readiness or permission to delete.
    """
    metadata = qdrant.get_collection(collection_name).config.metadata
    if metadata is None or INDEX_BUILD_MANIFEST_METADATA_KEY not in metadata:
        raise ValueError(f"Collection {collection_name!r} has no index build manifest metadata")

    try:
        manifest = IndexBuildManifest.model_validate(metadata[INDEX_BUILD_MANIFEST_METADATA_KEY])
        settings = IndexSettings.model_validate(metadata.get(INDEX_SETTINGS_METADATA_KEY))
    except ValidationError as exc:
        raise ValueError(
            f"Collection {collection_name!r} has invalid build manifest or index settings metadata"
        ) from exc

    if manifest.collection_name != collection_name or manifest.index_settings != settings:
        raise ValueError(
            f"Collection {collection_name!r} has inconsistent build manifest metadata"
        )
    return manifest


def get_collection_index_settings(
    qdrant: QdrantClient, collection_name: str
) -> IndexSettings:
    """Read and validate index settings stored in a collection's metadata.

    Missing or invalid settings raise IndexSettingsMetadataError. Qdrant errors
    propagate. This does not check whether the settings match the current
    configuration or whether indexing completed.
    """
    metadata = qdrant.get_collection(collection_name).config.metadata
    if metadata is None or INDEX_SETTINGS_METADATA_KEY not in metadata:
        raise IndexSettingsMetadataError(
            collection_name, "missing",
            f"Collection {collection_name!r} has no index settings metadata",
        )

    try:
        return IndexSettings.model_validate(metadata[INDEX_SETTINGS_METADATA_KEY])
    except ValidationError as exc:
        raise IndexSettingsMetadataError(
            collection_name, "invalid",
            f"Collection {collection_name!r} has invalid index settings metadata: {exc}",
        ) from exc


def compare_collection_index_settings(
    qdrant: QdrantClient, collection_name: str, current: IndexSettings
) -> list[str]:
    """Return field names differing from the collection's stored settings.

    Missing or invalid settings and Qdrant errors propagate. A match does not
    verify vector configuration, model weights, or indexing completion.
    """
    saved = get_collection_index_settings(qdrant, collection_name)
    return compare_index_settings(saved, current)


def ensure_collection_index_settings(
    qdrant: QdrantClient, collection_name: str, current: IndexSettings
) -> None:
    """Reject differing settings without modifying the collection.

    Differing settings raise IndexSettingsMismatchError. Missing or invalid
    settings and Qdrant errors propagate. This check does not verify vector
    configuration, model weights, or indexing completion.
    """
    differences = compare_collection_index_settings(qdrant, collection_name, current)
    if differences:
        raise IndexSettingsMismatchError(collection_name, differences)


def get_collection_index_status(qdrant: QdrantClient, collection_name: str) -> IndexStatus:
    """Return the index status; reject absent or unrecognized status metadata."""
    metadata = qdrant.get_collection(collection_name).config.metadata
    if metadata is None or INDEX_STATUS_METADATA_KEY not in metadata:
        raise ValueError(f"Collection {collection_name!r} has no index status metadata")
    status = metadata[INDEX_STATUS_METADATA_KEY]
    if status not in get_args(IndexStatus):
        raise ValueError(f"Collection {collection_name!r} has invalid index status metadata: {status!r}")
    return status


def ensure_collection_index_ready(qdrant: QdrantClient, collection_name: str) -> None:
    """Reject collections whose valid status is not ready."""
    if get_collection_index_status(qdrant, collection_name) != "ready":
        raise ValueError(f"Collection {collection_name!r} is still building its index")


def mark_collection_index_ready(
    qdrant: QdrantClient, collection_name: str, settings: IndexSettings
) -> None:
    """Mark a building index ready after the caller has completed all writes.

    The caller must exclusively own the indexing operation and only call this
    after successful ingestion. Status and settings checks are not atomic.
    If the update or verification fails, the resulting status may be unknown.
    """
    ensure_collection_index_settings(qdrant, collection_name, settings)
    if get_collection_index_status(qdrant, collection_name) != "building":
        raise ValueError(f"Collection {collection_name!r} index is not building")
    updated = qdrant.update_collection(
        collection_name=collection_name, metadata={INDEX_STATUS_METADATA_KEY: "ready"}
    )
    if not updated:
        raise RuntimeError(f"Qdrant did not confirm the index status update for {collection_name!r}")
    if get_collection_index_status(qdrant, collection_name) != "ready":
        raise RuntimeError(f"Collection {collection_name!r} index status update could not be verified")


def create_collection_with_settings(
    qdrant: QdrantClient,
    collection_name: str,
    embedding_dimension: int,
    settings: IndexSettings,
    *,
    manifest: IndexBuildManifest | None = None,
) -> None:
    """Create a new cosine collection with settings, status and optional manifest.

    Existing collections are not modified. Qdrant errors propagate. If creation
    succeeds but metadata cannot be verified, the new collection is left in
    place and the error must be resolved before using it. A manifest records
    caller-supplied identity but does not establish ownership.
    """
    if embedding_dimension <= 0:
        raise ValueError("embedding_dimension must be greater than zero")
    if manifest is not None and (
        manifest.collection_name != collection_name or manifest.index_settings != settings
    ):
        raise ValueError("Build manifest must match the collection name and index settings")

    metadata = {
        INDEX_SETTINGS_METADATA_KEY: settings.model_dump(),
        INDEX_STATUS_METADATA_KEY: "building",
    }
    if manifest is not None:
        metadata[INDEX_BUILD_MANIFEST_METADATA_KEY] = manifest.model_dump()
    created = qdrant.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=embedding_dimension,
            distance=Distance.COSINE,
        ),
        metadata=metadata,
    )
    if not created:
        raise RuntimeError(f"Qdrant did not create collection {collection_name!r}")

    stored_metadata = qdrant.get_collection(collection_name).config.metadata
    if not stored_metadata or any(
        stored_metadata.get(key) != value for key, value in metadata.items()
    ):
        raise RuntimeError(
            f"Collection {collection_name!r} was created but its index settings, "
            "status or build manifest metadata could not be verified"
        )


def ensure_collection_vector_config(
    qdrant: QdrantClient, collection_name: str, embedding_dimension: int
) -> None:
    """Validate an existing collection's unnamed cosine vector without modifying it."""
    if embedding_dimension <= 0:
        raise ValueError("embedding_dimension must be greater than zero")
    vectors = qdrant.get_collection(collection_name).config.params.vectors
    if not isinstance(vectors, VectorParams) or vectors.multivector_config is not None:
        raise ValueError(
            f"Collection {collection_name!r} must use a single unnamed dense vector"
        )
    if vectors.size != embedding_dimension or vectors.distance != Distance.COSINE:
        raise ValueError(
            f"Collection {collection_name!r} has vector size {vectors.size} and "
            f"distance {vectors.distance.value}; expected size {embedding_dimension} "
            f"and distance {Distance.COSINE.value}"
        )


def ensure_document_index_compatible(
    qdrant: QdrantClient,
    collection_name: str,
    embedding_dimension: int,
    settings: IndexSettings,
) -> None:
    """Check readiness, index settings and vector configuration without writing.

    The caller must ensure the collection is not modified concurrently, and
    that settings describe the actual embedding provider and model.
    """
    if embedding_dimension <= 0:
        raise ValueError("embedding_dimension must be greater than zero")
    ensure_collection_index_ready(qdrant, collection_name)
    ensure_collection_index_settings(qdrant, collection_name, settings)
    ensure_collection_vector_config(qdrant, collection_name, embedding_dimension)


def ensure_collection(
    qdrant: QdrantClient, collection_name: str, embedding_dimension: int
) -> bool:
    """
    Create a cosine-similarity collection or check its vector configuration.

    Args:
        qdrant (QdrantClient): Client connected to Qdrant.
        collection_name (str): Name of the collection.
        embedding_dimension (int): Size of vectors stored in the collection.

    Returns:
        bool: True if the collection was created, otherwise False.

    Raises:
        ValueError: If the dimension is not positive, or an existing
            collection does not have a compatible unnamed vector.
            Qdrant errors propagate.

    Matching dimensions and distance do not establish model compatibility.
    """
    if embedding_dimension <= 0:
        raise ValueError("embedding_dimension must be greater than zero")

    if qdrant.collection_exists(collection_name):
        ensure_collection_vector_config(qdrant, collection_name, embedding_dimension)
        return False

    qdrant.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=embedding_dimension,
            distance=Distance.COSINE,
        ),
    )
    return True
