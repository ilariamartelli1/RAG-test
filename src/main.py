import os
import pandas as pd

from torch import cuda
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from anthropic import Anthropic

if __package__:
    from .providers.anthropic_provider import AnthropicProvider
    from .storage.chunk_service import store_chunks
    from .storage.collection_service import ensure_collection
    from .processing.dataset_loader import load_dataset_dataframe
    from .processing.dataset_utils import chunk_dataset_row, row_to_text
    from .id_utils import group_positions_by_id
    from .chunk_models import ChunkPayload
    from .network_utils import is_online
    from .storage.qdrant_service import ensure_qdrant
    from .rag.rag_service import ask
else:
    from providers.anthropic_provider import AnthropicProvider
    from storage.chunk_service import store_chunks
    from storage.collection_service import ensure_collection
    from processing.dataset_loader import load_dataset_dataframe
    from processing.dataset_utils import chunk_dataset_row, row_to_text
    from id_utils import group_positions_by_id
    from chunk_models import ChunkPayload
    from network_utils import is_online
    from storage.qdrant_service import ensure_qdrant
    from rag.rag_service import ask

###############
## Utilities ##
###############

COLLECTION = "dataset_chunks"
BATCH_SIZE = 256
DATASET_NAME = "Hieu-Pham/kaggle_food_recipes"

LLM_MODEL = "claude-haiku-4-5"
EMBED_MODEL = "BAAI/bge-base-en-v1.5"

PROJECT_ROOT = os.path.dirname(
    os.path.abspath(__file__)
)

############
## Config ##
############

is_online()

load_dotenv()

ensure_qdrant(PROJECT_ROOT)

# Enable expandable GPU memory segments to reduce fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

#############
## Clients ##
#############

is_cuda_available = cuda.is_available()

embedder = SentenceTransformer(EMBED_MODEL,
    device="cuda" if is_cuda_available else "cpu")

if is_cuda_available:
    print(f"\nCUDA is available. Using GPU {cuda.get_device_name(0)} for embeddings.")
else:
    print("\nCUDA is not available. Using CPU for embeddings.")

embedding_dimension = embedder.get_embedding_dimension()
print(f"Sentence embedding dimension: {embedding_dimension}")

# how many texts the embedding model processes simultaneously internally
embedding_batch_size = 128 if is_cuda_available else 32
print(f"Using embedding batch size: {embedding_batch_size}")

qdrant = QdrantClient(
    host="localhost",
    port=6333,
)

claude = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

#########################
## Collection creation ##
#########################

if ensure_collection(qdrant, COLLECTION, embedding_dimension):
    print(f"\nCollection '{COLLECTION}' created")

###############
## Load data ##
###############

df = load_dataset_dataframe(
    DATASET_NAME,
    split="train",
    drop_columns=["Unnamed: 0", "Image_Name"],
)

df["full_text"] = df.apply(
    row_to_text,
    axis=1
)

print(f"\nQdrant status: {qdrant.get_collections()}")
print(f"{len(df)} recipes loaded. Will use embedding model: {EMBED_MODEL}")

####################
## Ingestion loop ##
####################

for i in range(0, len(df), BATCH_SIZE):
    batch_df: pd.DataFrame = df.iloc[i:i+BATCH_SIZE]
    chunks = [
        ChunkPayload(
            text=chunk.text,
            metadata={
                "source": DATASET_NAME,
                "row_id": chunk.row_id,
                "chunk_index": chunk.chunk_index,
                "field_name": chunk.field_name,
            },
        )
        for row_id, row in batch_df.iterrows()
        for chunk in chunk_dataset_row(row.drop(labels=["full_text"]), str(row_id))
    ]
    added = store_chunks(qdrant, embedder, COLLECTION, chunks, embedding_batch_size)
    print(f"Progress: {min(i+BATCH_SIZE, len(df))}/{len(df)} (+{added} new chunks)")

print(f"Done! Total chunks in collection '{COLLECTION}': {qdrant.count(COLLECTION).count}")

print("\nChecking for duplicate rows...")
duplicate_ids = group_positions_by_id(df["full_text"])
duplicates = {k: v for k, v in duplicate_ids.items() if len(v) > 1}

if duplicates:
    print(f"Found {len(duplicates)} duplicate hash IDs in dataset")
else:
    print("No duplicate hashes in dataset")

print(f"Dataset has {len(duplicate_ids)} unique hashes out of {len(df)} recipes")

###############################
## RAG Prompt Build Function ##
###############################

RAG_SYSTEM_PROMPT = """You are a helpful assistant.

You will be given relevant context retrieved from a database. Use ONLY this context to answer.
If the context doesn't contain enough information to answer, say so honestly.

Keep your answers concise and practical."""

ask(
    "I want to cook something with chicken and tomatoes. What do you suggest?",
    qdrant=qdrant,
    embedder=embedder,
    collection=COLLECTION,
    embedding_batch_size=embedding_batch_size,
    llm=AnthropicProvider(claude, LLM_MODEL),
    system_prompt=RAG_SYSTEM_PROMPT,
)
