"""Create or check a Qdrant collection for chunk embeddings."""

from typing import Literal

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from pydantic import ValidationError

if "." in __package__:
    from ..models.index_models import IndexSettings, compare_index_settings
else:
    from models.index_models import IndexSettings, compare_index_settings


INDEX_SETTINGS_METADATA_KEY = "index_settings"


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


def create_collection_with_settings(
    qdrant: QdrantClient,
    collection_name: str,
    embedding_dimension: int,
    settings: IndexSettings,
) -> None:
    """Create a new cosine collection and store its index settings in metadata.

    Existing collections are not modified. Qdrant errors propagate. If creation
    succeeds but metadata cannot be verified, the new collection is left in
    place and the error must be resolved before using it.
    """
    if embedding_dimension <= 0:
        raise ValueError("embedding_dimension must be greater than zero")

    metadata = {INDEX_SETTINGS_METADATA_KEY: settings.model_dump()}
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
    if not stored_metadata or stored_metadata.get(INDEX_SETTINGS_METADATA_KEY) != metadata[
        INDEX_SETTINGS_METADATA_KEY
    ]:
        raise RuntimeError(
            f"Collection {collection_name!r} was created but its index settings "
            "metadata could not be verified"
        )


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
        return False

    qdrant.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=embedding_dimension,
            distance=Distance.COSINE,
        ),
    )
    return True
