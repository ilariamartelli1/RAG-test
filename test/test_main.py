import importlib
import os
import runpy
import unittest
import warnings
from pathlib import Path
from unittest.mock import DEFAULT, MagicMock, patch

import pandas as pd


class MainTests(unittest.TestCase):
    def test_project_root_contains_docker_compose(self) -> None:
        main_module = importlib.import_module("src.main")

        self.assertEqual(Path(main_module.PROJECT_ROOT), Path(main_module.__file__).resolve().parent.parent)
        self.assertTrue((Path(main_module.PROJECT_ROOT) / "docker-compose.yml").is_file())

    @patch.dict(os.environ, {"PYTORCH_CUDA_ALLOC_CONF": "before"})
    @patch("src.network_utils.is_online")
    @patch("dotenv.load_dotenv")
    @patch("src.storage.qdrant_service.ensure_qdrant")
    @patch("src.processing.embedding_loader.load_sentence_transformer")
    @patch("src.processing.dataset_loader.load_dataset_dataframe")
    @patch("src.processing.dataset_ingestion.ingest_dataset")
    @patch("src.rag.rag_service.ask")
    @patch("qdrant_client.QdrantClient")
    @patch("anthropic.Anthropic")
    def test_import_does_not_run_pipeline(
        self, anthropic, qdrant, ask, ingest, load_dataset, load_model,
        ensure_qdrant, load_env, online,
    ) -> None:
        runpy.run_module("src.main", run_name="import_check")

        for operation in (online, load_env, ensure_qdrant, load_model, load_dataset,
                          ingest, ask, qdrant, anthropic):
            operation.assert_not_called()
        self.assertEqual(os.environ["PYTORCH_CUDA_ALLOC_CONF"], "before")

    @patch.dict(os.environ, {}, clear=False)
    @patch("src.network_utils.is_online")
    @patch("dotenv.load_dotenv")
    @patch("src.storage.qdrant_service.ensure_qdrant")
    @patch("src.processing.embedding_loader.load_sentence_transformer")
    @patch("qdrant_client.QdrantClient")
    @patch("anthropic.Anthropic")
    @patch("src.providers.anthropic_provider.AnthropicProvider")
    @patch("src.storage.collection_service.ensure_collection")
    @patch("src.processing.dataset_loader.load_dataset_dataframe")
    @patch("src.processing.dataset_ingestion.ingest_dataset")
    @patch("src.rag.rag_service.ask")
    @patch("builtins.print")
    def test_running_as_main_executes_pipeline(
        self, print_mock, ask, ingest, load_dataset, ensure_collection,
        provider, anthropic, qdrant, load_model, ensure_qdrant, load_env, online,
    ) -> None:
        dataframe = pd.DataFrame({
            "Title": ["Soup"], "Ingredients": ["Tomato"], "Instructions": ["Cook"],
        })
        embedder = MagicMock()
        embedder.get_embedding_dimension.return_value = 3
        load_model.return_value = (embedder, 32)
        load_dataset.return_value = dataframe
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="'src.main' found in sys.modules", category=RuntimeWarning
            )
            runpy.run_module("src.main", run_name="__main__")

        ask.assert_called_once()

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
        load_dataset_dataframe=DEFAULT,
        ingest_dataset=DEFAULT,
        ask=DEFAULT,
    )
    def test_main_runs_pipeline_with_existing_settings(self, print_mock, **mocks) -> None:
        main_module = importlib.import_module("src.main")
        dataframe = pd.DataFrame({
            "Title": ["Soup"], "Ingredients": ["Tomato"], "Instructions": ["Cook"],
        })
        embedder = MagicMock()
        embedder.get_embedding_dimension.return_value = 3
        qdrant = MagicMock()
        llm = MagicMock()
        provider = MagicMock()

        mocks["load_sentence_transformer"].return_value = (embedder, 32)
        mocks["QdrantClient"].return_value = qdrant
        mocks["Anthropic"].return_value = llm
        mocks["AnthropicProvider"].return_value = provider
        mocks["load_dataset_dataframe"].return_value = dataframe
        main_module.main()

        mocks["is_online"].assert_called_once_with()
        mocks["load_dotenv"].assert_called_once_with()
        mocks["ensure_qdrant"].assert_called_once_with(main_module.PROJECT_ROOT)
        mocks["ensure_collection"].assert_called_once_with(qdrant, main_module.COLLECTION, 3)
        mocks["load_dataset_dataframe"].assert_called_once_with(
            main_module.DATASET_NAME,
            split="train",
            drop_columns=["Unnamed: 0", "Image_Name"],
        )
        self.assertEqual(dataframe["full_text"].tolist(), [
            "Title: Soup\nIngredients: Tomato\nInstructions: Cook",
        ])
        mocks["ingest_dataset"].assert_called_once_with(
            dataframe, main_module.DATASET_NAME, qdrant, embedder,
            main_module.COLLECTION, main_module.BATCH_SIZE, 32,
        )
        mocks["ask"].assert_called_once_with(
            "I want to cook something with chicken and tomatoes. What do you suggest?",
            qdrant=qdrant, embedder=embedder, collection=main_module.COLLECTION,
            embedding_batch_size=32, llm=provider, system_prompt=main_module.RAG_SYSTEM_PROMPT,
        )


if __name__ == "__main__":
    unittest.main()
