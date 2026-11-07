from __future__ import annotations

import time
from typing import TYPE_CHECKING, Protocol

if __package__:
    from .chunk_service import retrieve_chunks
    from .prompt_utils import build_rag_prompt
else:
    from chunk_service import retrieve_chunks
    from prompt_utils import build_rag_prompt

if TYPE_CHECKING:
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer


class LLMProvider(Protocol):
    """Interface for generating an answer from messages and instructions."""

    def generate(self, messages: list[dict[str, str]], system_prompt: str) -> str:
        ...


def ask(
    question: str,
    *,
    qdrant: QdrantClient,
    embedder: SentenceTransformer,
    collection: str,
    embedding_batch_size: int,
    llm: LLMProvider,
    system_prompt: str,
    retries: int = 3,
) -> str:
    """
    Retrieve context and ask the language model.

    Args:
        question (str): User question.
        qdrant (QdrantClient): Client connected to Qdrant.
        embedder (SentenceTransformer): Model used to embed the question.
        collection (str): Qdrant collection to search.
        embedding_batch_size (int): Batch size used by the embedding model.
        llm (LLMProvider): Provider used to generate the answer.
        system_prompt (str): Instructions supplied to the language model.
        retries (int): Number of attempts to generate an answer.

    Returns:
        str: Language model answer.
    """
    results = retrieve_chunks(qdrant, embedder, collection, question, embedding_batch_size)
    print(f"\nRetrieved {len(results)} chunks:")
    for chunk, score in results:
        print(f"  {score:.4f}  {chunk.metadata}")
    print()

    messages = build_rag_prompt(question, [chunk.text for chunk, _ in results])

    for attempt in range(retries):
        try:
            response = llm.generate(messages=messages, system_prompt=system_prompt)
            break
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            if attempt == retries - 1:
                raise
            time.sleep(1)

    answer = response.strip()
    print(f"Answer:\n{answer}")
    return answer
