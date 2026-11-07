import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from anthropic import Anthropic

if __package__:
    from .providers.anthropic_provider import AnthropicProvider
    from .storage.collection_service import ensure_collection
    from .processing.dataset_ingestion import ingest_dataset
    from .processing.dataset_loader import load_dataset_dataframe
    from .processing.dataset_utils import row_to_text
    from .processing.embedding_loader import load_sentence_transformer
    from .id_utils import group_positions_by_id
    from .network_utils import is_online
    from .storage.qdrant_service import ensure_qdrant
    from .rag.rag_service import ask
else:
    from providers.anthropic_provider import AnthropicProvider
    from storage.collection_service import ensure_collection
    from processing.dataset_ingestion import ingest_dataset
    from processing.dataset_loader import load_dataset_dataframe
    from processing.dataset_utils import row_to_text
    from processing.embedding_loader import load_sentence_transformer
    from id_utils import group_positions_by_id
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

embedder, embedding_batch_size = load_sentence_transformer(EMBED_MODEL)

embedding_dimension = embedder.get_embedding_dimension()
print(f"Sentence embedding dimension: {embedding_dimension}")

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

ingest_dataset(df, DATASET_NAME, qdrant, embedder, COLLECTION, BATCH_SIZE, embedding_batch_size)

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
