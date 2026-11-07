from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


def ensure_collection(
    qdrant: QdrantClient, collection_name: str, embedding_dimension: int
) -> bool:
    """
    Create a cosine-similarity collection if it does not already exist.

    Args:
        qdrant (QdrantClient): Client connected to Qdrant.
        collection_name (str): Name of the collection.
        embedding_dimension (int): Size of vectors stored in the collection.

    Returns:
        bool: True if the collection was created, otherwise False.
    """
    if qdrant.collection_exists(collection_name):
        return False

    qdrant.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=embedding_dimension,
            distance=Distance.COSINE,
        ),
    )
    return True
