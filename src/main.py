import os
import sys
import time
import socket
import hashlib
import subprocess
import pandas as pd

from typing import Any, List
from torch import cuda
from dotenv import load_dotenv
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from anthropic import Anthropic
from pydantic import BaseModel

###############
## Utilities ##
###############

COLLECTION = "recipes"
BATCH_SIZE = 256

LLM_MODEL = "claude-haiku-4-5"
EMBED_MODEL = "BAAI/bge-base-en-v1.5"

PROJECT_ROOT = os.path.dirname(
    os.path.abspath(__file__)
)

class Recipe(BaseModel):
    """
    Pydantic model for recipe payloads.
    """
    title: str
    ingredients: str
    instructions: str

def is_online() -> None:
    """
    Check if the machine is online by connecting to a public DNS server.
    Exits the program if offline.
    """
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=5)
    except OSError as e:
        print(f"Offline or DNS/network issue detected:\n{e}")
        sys.exit(1)

def is_qdrant_running() -> bool:
    """
    Check if Qdrant is running locally on port 6333.
    Returns:
        bool: True if Qdrant is running, False otherwise.
    """
    try:
        socket.create_connection(
            ("localhost", 6333),
            timeout=2,
        )
        return True
    except OSError:
        return False

def ensure_qdrant() -> None:
    """
    Ensure Qdrant is running, start it with Docker Compose if not.
    """
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

def safe_str(val: Any) -> str:
    """
    Convert a value to string, handling NaN/None.
    Args:
        val: Value to convert.
    Returns:
        str: String representation or empty string.
    """
    if pd.isna(val):
        return ""
    return str(val)

def recipe_to_text(row: pd.Series) -> str:
    """
    Convert a recipe row to a formatted text string.
    Args:
        row (pd.Series): Recipe row.
    Returns:
        str: Formatted recipe text.
    """
    title = safe_str(row["Title"])
    ingredients = safe_str(row["Ingredients"])
    instructions = safe_str(row["Instructions"])
    return f"Title: {title}\nIngredients: {ingredients}\nInstructions: {instructions}".strip()

def make_id(text: str) -> int:
    """
    Generate a numeric ID from MD5 hash.
    Args:
        text (str): Input text.
    Returns:
        int: 64-bit integer ID.
    """
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

####################
## Ingestion loop ##
####################

for i in range(0, len(df), BATCH_SIZE):
    batch_df: pd.DataFrame = df.iloc[i:i+BATCH_SIZE]
    texts: list[str] = batch_df["full_text"].tolist()

    # Generate IDs for all texts first
    ids: list[int] = [make_id(text) for text in texts]

    # Check which ones already exist in Qdrant
    try:
        existing_points = qdrant.retrieve(
            collection_name=COLLECTION,
            ids=ids,
        )
        existing_ids: set[int] = {point.id for point in existing_points}
    except Exception as e:
        # If retrieve fails, assume none exist (will try to insert)
        existing_ids: set[int] = set()
        print(f"Warning: retrieve() error at batch {i//BATCH_SIZE}, assuming 0 exist: {e}")

    # Filter to only texts that don't exist yet
    new_indices: list[int] = [j for j, id in enumerate(ids) if id not in existing_ids]

    if not new_indices:
        # All recipes already exist, skip embedding
        print(f"{len(batch_df)} recipes already in collection, skipping...")
        continue

    # Only encode the texts that need to be inserted
    texts_to_encode: list[str] = [texts[j] for j in new_indices]

    embeddings: list[list[float]] = embedder.encode(
        texts_to_encode,
        batch_size=embedding_batch_size,
        show_progress_bar=False,
    ).tolist()

    points: list[PointStruct] = []

    for embed_idx, data_idx in enumerate(new_indices):
        row = batch_df.iloc[data_idx]
        text = texts[data_idx]

        # Use Pydantic model for payload validation
        recipe_payload = Recipe(
            title=safe_str(row.Title),
            ingredients=safe_str(row.Ingredients),
            instructions=safe_str(row.Instructions),
        ).model_dump()

        point = PointStruct(
            id=ids[data_idx],
            vector=embeddings[embed_idx],
            payload=recipe_payload
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

def retrieve_recipes(query: str, top_n: int = 5) -> List[dict]:
    """
    Retrieve the most relevant recipes for a given query.
    Args:
        query (str): User query.
        top_n (int): Number of top results to return.
    Returns:
        list: List of recipes as dicts with title, ingredients, instructions, and score.
    """
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
    
    # Step 3: Format results using Pydantic validation
    recipes: List[dict] = []
    for point in results.points:
        # Validate payload with Recipe model
        recipe = Recipe(**point.payload)
        recipes.append({
            "title": recipe.title,
            "ingredients": recipe.ingredients,
            "instructions": recipe.instructions,
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

def message_with_context(question: str, context: str) -> str:
    """
    Build the user message with context for the LLM prompt.
    Args:
        question (str): User question.
        context (str): Retrieved recipes context.
    Returns:
        str: Formatted user message.
    """
    user_message = f"""Here are the most relevant recipes from the database:

{context}

---

User question: {question}"""
    return user_message


def build_rag_prompt(question: str, recipes: list[dict]) -> list[dict]:
    """
    Build the messages list for the RAG call.
    Args:
        question (str): User question.
        recipes (list): List of recipe dicts.
    Returns:
        list: List of message dicts for the LLM.
    """
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

def ask(question: str, retries: int = 3) -> str:
    """
    Full RAG pipeline: retrieve recipes, then ask the LLM.
    Args:
        question (str): User question.
        retries (int): Number of retries for LLM call.
    Returns:
        str: LLM answer.
    """
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
