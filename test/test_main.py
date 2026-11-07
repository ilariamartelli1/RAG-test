import importlib
import os
import runpy
import unittest
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd


class MainTests(unittest.TestCase):
    def test_project_root_contains_docker_compose(self) -> None:
        main_module = importlib.import_module("src.main")

        self.assertEqual(Path(main_module.PROJECT_ROOT), Path(main_module.__file__).resolve().parent.parent)
        self.assertTrue((Path(main_module.PROJECT_ROOT) / "docker-compose.yml").is_file())

    def test_import_does_not_run_pipeline(self) -> None:
        with (
            patch("src.network_utils.is_online") as online,
            patch("dotenv.load_dotenv") as load_env,
            patch("src.storage.qdrant_service.ensure_qdrant") as ensure_qdrant,
            patch("src.processing.embedding_loader.load_sentence_transformer") as load_model,
            patch("src.processing.dataset_loader.load_dataset_dataframe") as load_dataset,
            patch("src.processing.dataset_ingestion.ingest_dataset") as ingest,
            patch("src.rag.rag_service.ask") as ask,
            patch("qdrant_client.QdrantClient") as qdrant,
            patch("anthropic.Anthropic") as anthropic,
            patch.dict(os.environ, {"PYTORCH_CUDA_ALLOC_CONF": "before"}),
        ):
            runpy.run_module("src.main", run_name="import_check")

            for operation in (online, load_env, ensure_qdrant, load_model, load_dataset,
                              ingest, ask, qdrant, anthropic):
                operation.assert_not_called()
            self.assertEqual(os.environ["PYTORCH_CUDA_ALLOC_CONF"], "before")

    def test_running_as_main_executes_pipeline(self) -> None:
        dataframe = pd.DataFrame({
            "Title": ["Soup"], "Ingredients": ["Tomato"], "Instructions": ["Cook"],
        })
        embedder = MagicMock()
        embedder.get_embedding_dimension.return_value = 3
        with (
            patch("src.network_utils.is_online"),
            patch("dotenv.load_dotenv"),
            patch("src.storage.qdrant_service.ensure_qdrant"),
            patch("src.processing.embedding_loader.load_sentence_transformer",
                  return_value=(embedder, 32)),
            patch("qdrant_client.QdrantClient"),
            patch("anthropic.Anthropic"),
            patch("src.providers.anthropic_provider.AnthropicProvider"),
            patch("src.storage.collection_service.ensure_collection"),
            patch("src.processing.dataset_loader.load_dataset_dataframe", return_value=dataframe),
            patch("src.processing.dataset_ingestion.ingest_dataset"),
            patch("src.rag.rag_service.ask") as ask,
            patch("builtins.print"),
            patch.dict(os.environ, {}, clear=False),
            warnings.catch_warnings(),
        ):
            warnings.filterwarnings(
                "ignore", message="'src.main' found in sys.modules", category=RuntimeWarning
            )
            runpy.run_module("src.main", run_name="__main__")

        ask.assert_called_once()

    def test_main_runs_pipeline_with_existing_settings(self) -> None:
        main_module = importlib.import_module("src.main")
        dataframe = pd.DataFrame({
            "Title": ["Soup"], "Ingredients": ["Tomato"], "Instructions": ["Cook"],
        })
        embedder = MagicMock()
        embedder.get_embedding_dimension.return_value = 3
        qdrant = MagicMock()
        llm = MagicMock()
        provider = MagicMock()

        with (
            patch.object(main_module, "is_online") as online,
            patch.object(main_module, "load_dotenv") as load_env,
            patch.object(main_module, "ensure_qdrant") as ensure_qdrant,
            patch.object(main_module, "load_sentence_transformer", return_value=(embedder, 32)),
            patch.object(main_module, "QdrantClient", return_value=qdrant),
            patch.object(main_module, "Anthropic", return_value=llm),
            patch.object(main_module, "AnthropicProvider", return_value=provider),
            patch.object(main_module, "ensure_collection") as ensure_collection,
            patch.object(main_module, "load_dataset_dataframe", return_value=dataframe) as load_dataset,
            patch.object(main_module, "ingest_dataset") as ingest,
            patch.object(main_module, "ask") as ask,
            patch("builtins.print"),
            patch.dict(os.environ, {}, clear=False),
        ):
            main_module.main()

        online.assert_called_once_with()
        load_env.assert_called_once_with()
        ensure_qdrant.assert_called_once_with(main_module.PROJECT_ROOT)
        ensure_collection.assert_called_once_with(qdrant, main_module.COLLECTION, 3)
        load_dataset.assert_called_once_with(
            main_module.DATASET_NAME,
            split="train",
            drop_columns=["Unnamed: 0", "Image_Name"],
        )
        self.assertEqual(dataframe["full_text"].tolist(), [
            "Title: Soup\nIngredients: Tomato\nInstructions: Cook",
        ])
        ingest.assert_called_once_with(
            dataframe, main_module.DATASET_NAME, qdrant, embedder,
            main_module.COLLECTION, main_module.BATCH_SIZE, 32,
        )
        ask.assert_called_once_with(
            "I want to cook something with chicken and tomatoes. What do you suggest?",
            qdrant=qdrant, embedder=embedder, collection=main_module.COLLECTION,
            embedding_batch_size=32, llm=provider, system_prompt=main_module.RAG_SYSTEM_PROMPT,
        )


if __name__ == "__main__":
    unittest.main()
