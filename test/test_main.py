"""Tests for the example entry point and import safety."""

import importlib
import os
import runpy
import unittest
import warnings
from pathlib import Path
from unittest.mock import DEFAULT, MagicMock, patch

import pandas as pd

from src.models.chunk_models import ChunkPayload
from src.loaders.dataset_loader import dataframe_to_records
from src.chunkers.dataset_chunker import chunk_dataset_row
from src.utils.dataset_utils import row_to_text
from src.models.record_models import SourceRecord


class MainTests(unittest.TestCase):
    def test_project_root_contains_docker_compose(self) -> None:
        main_module = importlib.import_module("src.main")

        self.assertEqual(Path(main_module.PROJECT_ROOT), Path(main_module.__file__).resolve().parent.parent)
        self.assertTrue((Path(main_module.PROJECT_ROOT) / "docker-compose.yml").is_file())

    @patch.dict(os.environ, {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False"})
    @patch("src.utils.network_utils.is_online")
    @patch("dotenv.load_dotenv")
    @patch("src.storage.qdrant_service.ensure_qdrant")
    @patch("src.embeddings.embedding_loader.load_sentence_transformer")
    @patch("src.loaders.dataset_loader.load_dataset_records")
    @patch("src.ingestion.ingestion_service.ingest_records")
    @patch("src.storage.chunk_service.store_chunks")
    @patch("src.rag.rag_service.ask")
    @patch("qdrant_client.QdrantClient")
    @patch("anthropic.Anthropic")
    def test_import_does_not_run_pipeline(
        self, anthropic, qdrant, ask, store, ingest, load_dataset, load_model,
        ensure_qdrant, load_env, online,
    ) -> None:
        runpy.run_module("src.main", run_name="import_check")

        for operation in (online, load_env, ensure_qdrant, load_model, load_dataset,
                          ingest, store, ask, qdrant, anthropic):
            operation.assert_not_called()
        self.assertEqual(os.environ["PYTORCH_CUDA_ALLOC_CONF"], "expandable_segments:False")

    @patch.dict(os.environ, {}, clear=False)
    @patch("src.utils.network_utils.is_online")
    @patch("dotenv.load_dotenv")
    @patch("src.storage.qdrant_service.ensure_qdrant")
    @patch("src.embeddings.embedding_loader.load_sentence_transformer")
    @patch("qdrant_client.QdrantClient")
    @patch("anthropic.Anthropic")
    @patch("src.providers.anthropic_provider.AnthropicProvider")
    @patch("src.storage.collection_service.ensure_collection")
    @patch("src.loaders.dataset_loader.load_dataset_records")
    @patch("src.ingestion.ingestion_service.ingest_records")
    @patch("src.rag.rag_service.ask")
    @patch("builtins.print")
    def test_running_as_main_executes_pipeline(
        self, print_mock, ask, ingest, load_dataset, ensure_collection,
        provider, anthropic, qdrant, load_model, ensure_qdrant, load_env, online,
    ) -> None:
        records = [SourceRecord(
            content={"Title": "Soup", "Ingredients": "Tomato", "Instructions": "Cook"},
            metadata={"source": "dataset", "row_id": "0"},
        )]
        embedder = MagicMock()
        embedder.get_embedding_dimension.return_value = 3
        load_model.return_value = (embedder, 32)
        load_dataset.return_value = records
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="'src.main' found in sys.modules", category=RuntimeWarning
            )
            runpy.run_module("src.main", run_name="__main__")

        ask.assert_called_once()
        print_mock.assert_any_call("No duplicate hashes in dataset")

    @patch.dict(os.environ, {}, clear=False)
    @patch("builtins.print")
    @patch.multiple(
        "src.main",
        is_online=DEFAULT,
        load_dotenv=DEFAULT,
        ensure_qdrant=DEFAULT,
        load_sentence_transformer=DEFAULT,
        QdrantClient=DEFAULT,
        Anthropic=DEFAULT,
        AnthropicProvider=DEFAULT,
        ensure_collection=DEFAULT,
        load_dataset_records=DEFAULT,
        store_chunks=DEFAULT,
        BATCH_SIZE=2,
        ask=DEFAULT,
    )
    def test_main_runs_pipeline_with_existing_settings(self, print_mock, **mocks) -> None:
        main_module = importlib.import_module("src.main")
        dataframe = pd.DataFrame({
            "Title": ["Soup", "Soup", "x" * 1100],
            "Servings": [2, 2, 4],
        })
        records = dataframe_to_records(dataframe, main_module.DATASET_NAME)
        embedder = MagicMock()
        embedder.get_embedding_dimension.return_value = 3
        qdrant = MagicMock()
        llm = MagicMock()
        provider = MagicMock()

        mocks["load_sentence_transformer"].return_value = (embedder, 32)
        mocks["QdrantClient"].return_value = qdrant
        mocks["Anthropic"].return_value = llm
        mocks["AnthropicProvider"].return_value = provider
        mocks["load_dataset_records"].return_value = records
        mocks["store_chunks"].return_value = 2
        main_module.main()

        mocks["is_online"].assert_called_once_with()
        mocks["load_dotenv"].assert_called_once_with()
        mocks["ensure_qdrant"].assert_called_once_with(main_module.PROJECT_ROOT)
        mocks["ensure_collection"].assert_called_once_with(qdrant, main_module.COLLECTION, 3)
        mocks["load_dataset_records"].assert_called_once_with(
            main_module.DATASET_NAME,
            split="train",
            drop_columns=["Unnamed: 0", "Image_Name"],
        )
        batches = [call.args[3] for call in mocks["store_chunks"].call_args_list]
        self.assertEqual([len(batch) for batch in batches], [2, 2, 1])
        dataframe["full_text"] = dataframe.apply(row_to_text, axis=1)
        old_chunks = [
            ChunkPayload(
                text=chunk.text,
                metadata={
                    "source": main_module.DATASET_NAME,
                    "row_id": chunk.row_id,
                    "chunk_index": chunk.chunk_index,
                    "field_name": chunk.field_name,
                },
            )
            for row_id, row in dataframe.iterrows()
            for chunk in chunk_dataset_row(row.drop(labels=["full_text"]), str(row_id))
        ]
        self.assertEqual(
            [chunk for batch in batches for chunk in batch],
            old_chunks,
        )
        for call in mocks["store_chunks"].call_args_list:
            self.assertEqual(call.args[:3], (qdrant, embedder, main_module.COLLECTION))
            self.assertEqual(call.kwargs, {"embedding_batch_size": 32})
        print_mock.assert_any_call("Found 1 duplicate hash IDs in dataset")
        print_mock.assert_any_call("Dataset has 2 unique hashes out of 3 recipes")
        mocks["ask"].assert_called_once_with(
            "I want to cook something with chicken and tomatoes. What do you suggest?",
            qdrant=qdrant, embedder=embedder, collection=main_module.COLLECTION,
            embedding_batch_size=32, llm=provider, system_prompt=main_module.RAG_SYSTEM_PROMPT,
        )


if __name__ == "__main__":
    unittest.main()
