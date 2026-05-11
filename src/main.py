import os
import sys
import time
import socket
import hashlib
import subprocess
import pandas as pd

from torch import cuda
from dotenv import load_dotenv
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from anthropic import Anthropic

###############
## Utilities ##
###############

# CHECKPOINT_FILE = "ingestion_checkpoint.txt"
COLLECTION = "recipes"
BATCH_SIZE = 256

LLM_MODEL = "claude-haiku-4-5"
EMBED_MODEL = "BAAI/bge-base-en-v1.5"

PROJECT_ROOT = os.path.dirname(
    os.path.abspath(__file__)
)

def is_online():
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=5)
    except OSError as e:
        print(f"Offline or DNS/network issue detected:\n{e}")
        sys.exit(1)

def is_qdrant_running():
    try:
        socket.create_connection(
            ("localhost", 6333),
            timeout=2,
        )
        return True
    except OSError:
        return False

def ensure_qdrant():
    if is_qdrant_running():
        print("Qdrant already running")
        return
    print("Starting Qdrant...")

    try:
        subprocess.run(
            ["docker", "compose", "up", "-d"],
            cwd=PROJECT_ROOT,
            check=True,
        )
        print("Qdrant started")
    except subprocess.CalledProcessError as e:
        print(f"Failed to start docker compose:\n{e}")
        sys.exit(1)

def safe_str(val):
    """Convert to string, handling NaN/None."""
    if pd.isna(val):
        return ""
    return str(val)

def recipe_to_text(row):
    title = safe_str(row["Title"])
    ingredients = safe_str(row["Ingredients"])
    instructions = safe_str(row["Instructions"])
    return f"Title: {title}\nIngredients: {ingredients}\nInstructions: {instructions}".strip()

def make_id(text):
    """Generate a numeric ID from MD5 hash."""
    hash_hex = hashlib.md5(text.encode()).hexdigest()
    # Convert first 16 hex characters to a 64-bit integer
    return int(hash_hex[:16], 16)

############
## Config ##
############

is_online()

load_dotenv()

ensure_qdrant()

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

collection_exists = qdrant.collection_exists(COLLECTION)
if not collection_exists:
    qdrant.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(
            size=embedding_dimension,
            distance=Distance.COSINE,
        ),
    )
    print(f"\nCollection '{COLLECTION}' created")

###############
## Load data ##
###############

ds = load_dataset(
    "Hieu-Pham/kaggle_food_recipes",
    split="train"
)

df = ds.to_pandas().drop(
    columns=["Unnamed: 0", "Image_Name"]
)

df["full_text"] = df.apply(
    recipe_to_text,
    axis=1
)

print(f"\nQdrant status: {qdrant.get_collections()}")
print(f"{len(df)} recipes loaded. Will use embedding model: {EMBED_MODEL}")

#########################
## Checkpoint recovery ##
#########################

# Always start from 0 - we'll skip recipes that are already in Qdrant
# This is safer than relying on checkpoint which might be out of sync
start_idx = 0

# if collection_exists:
#     if os.path.exists(CHECKPOINT_FILE):
#         with open(CHECKPOINT_FILE, "r") as f:
#             try:
#                 last_idx = int(f.read().strip())
#                 print(f"\nCheckpoint found at index {last_idx}. Will verify which recipes are already in Qdrant.")
#             except ValueError:
#                 print(f"\nInvalid checkpoint file. Starting from index 0.")

print(f"Starting ingestion from index {start_idx}. Will skip recipes already in Qdrant.\n")

####################
## Ingestion loop ##
####################

for i in range(start_idx, len(df), BATCH_SIZE):

    batch_df = df.iloc[i:i+BATCH_SIZE]
    texts = batch_df["full_text"].tolist()

    # Generate IDs for all texts first
    ids = [make_id(text) for text in texts]
    
    # Check which ones already exist in Qdrant
    try:
        existing_points = qdrant.retrieve(
            collection_name=COLLECTION,
            ids=ids,
        )
        existing_ids = {point.id for point in existing_points}
    except Exception as e:
        # If retrieve fails, assume none exist (will try to insert)
        existing_ids = set()
        print(f"Warning: retrieve() error at batch {i//BATCH_SIZE}, assuming 0 exist: {e}")

    # Filter to only texts that don't exist yet
    new_indices = [j for j, id in enumerate(ids) if id not in existing_ids]
    
    if not new_indices:
        # All recipes already exist, skip embedding
        print(f"{len(batch_df)} recipes already in collection, skipping...")
        continue
    
    # Only encode the texts that need to be inserted
    texts_to_encode = [texts[j] for j in new_indices]

    embeddings = embedder.encode(
        texts_to_encode,
        batch_size=embedding_batch_size,
        show_progress_bar=False,
    ).tolist()

    points = []

    for embed_idx, data_idx in enumerate(new_indices):
        row = batch_df.iloc[data_idx]
        text = texts[data_idx]

        point = PointStruct(
            id=ids[data_idx],
            vector=embeddings[embed_idx],
            payload={
                "title": safe_str(row.Title),
                "ingredients": safe_str(row.Ingredients),
                "instructions": safe_str(row.Instructions),
            }
        )

        points.append(point)
    
    try:
        qdrant.upsert(collection_name=COLLECTION, points=points)
        print(f"Progress: {min(i+BATCH_SIZE, len(df))}/{len(df)} (+{len(points)} new)")
    except Exception as e:
        print(f"ERROR upserting points: {e}")
        print(f"First point ID: {points[0].id if points else 'No points'}")
        print(f"First point vector length: {len(points[0].vector) if points else 0}")
        raise

    # with open(CHECKPOINT_FILE, "w") as f:
    #     f.write(str(i + len(batch_df)))

print(f"Done! Total recipes in collection '{COLLECTION}': {qdrant.count(COLLECTION).count}")

# Verify which recipes are missing
print("\nVerifying ingestion...")
# Check for duplicate hashes in the dataset
all_ids = []
duplicate_ids = {}

for i in range(len(df)):
    text = df.iloc[i]["full_text"]
    id = make_id(text)
    all_ids.append(id)
    
    if id in duplicate_ids:
        duplicate_ids[id].append(i)
    else:
        duplicate_ids[id] = [i]

duplicates = {k: v for k, v in duplicate_ids.items() if len(v) > 1}

if duplicates:
    print(f"Found {len(duplicates)} duplicate hash IDs in dataset")
else:
    print("No duplicate hashes in dataset")

print(f"Dataset has {len(set(all_ids))} unique hashes out of {len(df)} recipes")

###############################
## Recipe Retrieval Function ##
###############################

def retrieve_recipes(query, top_n=5):
    """Retrieve the most relevant recipes for a given query."""
    # Step 1: Query embedding
    query_vector = embedder.encode(
        query,
        batch_size=embedding_batch_size,
        show_progress_bar=False,
    ).tolist()
    
    # Step 2: Search in Qdrant
    results = qdrant.query_points(
        collection_name=COLLECTION,
        query=query_vector,
        limit=top_n,
        with_payload=True,
    )
    
    # Step 3: Format results
    recipes = []
    for point in results.points:
        recipes.append({
            "title": point.payload["title"],
            "ingredients": point.payload["ingredients"],
            "instructions": point.payload["instructions"],
            "score": round(point.score, 4),
        })
    return recipes

###############################
## RAG Prompt Build Function ##
###############################

RAG_SYSTEM_PROMPT = """You are a helpful cooking assistant. You answer questions about recipes.

You will be given a set of recipes retrieved from a database. Use ONLY these recipes to answer.
If the retrieved recipes don't contain enough information to answer, say so honestly.

When suggesting a recipe, mention its name and briefly describe why it fits the user's request.
Keep your answers concise and practical."""

def message_with_context(question, context):
    user_message = f"""Here are the most relevant recipes from the database:

{context}

---

User question: {question}"""
    return user_message


def build_rag_prompt(question, recipes):
    """Build the messages list for the RAG call."""
    # Format retrieved recipes as context
    context = "\n\n---\n\n".join([
        f"**{r['title']}** (relevance: {r['score']})\n"
        f"Ingredients: {r['ingredients']}\n"
        f"Instructions: {r['instructions']}"
        for r in recipes
    ])
    
    user_message = message_with_context(question, context)
    
    return [{"role": "user", "content": user_message}]

##################
## Ask Function ##
##################

def ask(question, retries=3):
    """Full RAG pipeline: retrieve recipes, then ask the LLM."""
    # 1. Retrieve
    recipes = retrieve_recipes(question)
    print(f"\nRetrieved {len(recipes)} recipes:")
    for r in recipes:
        print(f"  {r['score']:.4f}  {r['title']}")
    print()
    
    # 2. Build the prompt
    messages = build_rag_prompt(question, recipes)
    
    # 3. Generate the response
    for attempt in range(retries):
        try:
            response = claude.messages.create(
                model=LLM_MODEL,
                max_tokens=1024,
                system=RAG_SYSTEM_PROMPT,
                messages=messages
            )
            break
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            if attempt == retries - 1:
                raise
            time.sleep(1)  # wait before retrying

    answer = response.content[0].text.strip()
    print(f"Answer:\n{answer}")
    return answer

ask("I want to cook something with chicken and tomatoes. What do you suggest?")
