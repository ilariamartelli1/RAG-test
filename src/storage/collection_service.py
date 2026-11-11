"""Create or check a Qdrant collection for chunk embeddings."""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

if "." in __package__:
    from ..models.index_models import IndexSettings
else:
    from models.index_models import IndexSettings


INDEX_SETTINGS_METADATA_KEY = "index_settings"


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
