from sentence_transformers import SentenceTransformer
from torch import cuda


def load_sentence_transformer(model_name: str) -> tuple[SentenceTransformer, int]:
    """
    Load a SentenceTransformers model and select its embedding batch size.

    Args:
        model_name (str): Name or path of the embedding model.

    Returns:
        tuple[SentenceTransformer, int]: Loaded model and embedding batch size.
    """
    is_cuda_available = cuda.is_available()
    embedder = SentenceTransformer(
        model_name, device="cuda" if is_cuda_available else "cpu"
    )

    if is_cuda_available:
        print(f"\nCUDA is available. Using GPU {cuda.get_device_name(0)} for embeddings.")
    else:
        print("\nCUDA is not available. Using CPU for embeddings.")

    return embedder, 128 if is_cuda_available else 32
