"""Create or check a Qdrant collection for chunk embeddings."""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


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
