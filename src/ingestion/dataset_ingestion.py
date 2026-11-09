"""Obsolete DataFrame-based ingestion retained for compatibility.

The active pipeline uses ``ingestion_service.ingest_records`` instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if "." in __package__:
    from ..models.chunk_models import ChunkPayload
    from ..chunkers.dataset_chunker import chunk_dataset_row
    from ..storage.chunk_service import store_chunks
    from ..embeddings.sentence_transformer_provider import SentenceTransformerProvider
else:
    from models.chunk_models import ChunkPayload
    from chunkers.dataset_chunker import chunk_dataset_row
    from storage.chunk_service import store_chunks
    from embeddings.sentence_transformer_provider import SentenceTransformerProvider

if TYPE_CHECKING:
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer


def ingest_dataset(
    dataframe: pd.DataFrame,
    source: str,
    qdrant: QdrantClient,
    embedder: SentenceTransformer,
    collection: str,
    batch_size: int,
    embedding_batch_size: int,
) -> None:
    """
    Chunk and store dataset rows in batches.

    Args:
        dataframe (pd.DataFrame): Dataset with a derived ``full_text`` column.
        source (str): Dataset source recorded in chunk metadata.
        qdrant (QdrantClient): Client connected to Qdrant.
        embedder (SentenceTransformer): Model used to embed chunks.
        collection (str): Target Qdrant collection.
        batch_size (int): Number of dataset rows processed per batch.
        embedding_batch_size (int): Batch size used by the embedding model.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    embedding_provider = SentenceTransformerProvider(embedder)
    for i in range(0, len(dataframe), batch_size):
        batch_df = dataframe.iloc[i:i + batch_size]
        chunks = [
            ChunkPayload(
                text=chunk.text,
                metadata={
                    "source": source,
                    "row_id": chunk.row_id,
                    "chunk_index": chunk.chunk_index,
                    "field_name": chunk.field_name,
                },
            )
            for row_id, row in batch_df.iterrows()
            for chunk in chunk_dataset_row(row.drop(labels=["full_text"]), str(row_id))
        ]
        added = store_chunks(qdrant, embedding_provider, collection, chunks, embedding_batch_size)
        print(f"Progress: {min(i + batch_size, len(dataframe))}/{len(dataframe)} (+{added} new chunks)")
