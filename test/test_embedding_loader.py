import unittest
from unittest.mock import patch

from src.embeddings.embedding_loader import load_sentence_transformer


class EmbeddingLoaderTests(unittest.TestCase):
    @patch("src.embeddings.embedding_loader.SentenceTransformer")
    @patch("src.embeddings.embedding_loader.cuda.is_available", return_value=False)
    @patch("builtins.print")
    def test_loads_cpu_model(self, print_mock, is_available, model_class) -> None:
        model, batch_size = load_sentence_transformer("example-model")

        model_class.assert_called_once_with("example-model", device="cpu")
        self.assertIs(model, model_class.return_value)
        self.assertEqual(batch_size, 32)
        print_mock.assert_called_once_with("\nCUDA is not available. Using CPU for embeddings.")

    @patch("src.embeddings.embedding_loader.SentenceTransformer")
    @patch("src.embeddings.embedding_loader.cuda.is_available", return_value=True)
    @patch("src.embeddings.embedding_loader.cuda.get_device_name", return_value="Example GPU")
    @patch("builtins.print")
    def test_loads_cuda_model(self, print_mock, device_name, is_available, model_class) -> None:
        model, batch_size = load_sentence_transformer("example-model")

        model_class.assert_called_once_with("example-model", device="cuda")
        device_name.assert_called_once_with(0)
        self.assertIs(model, model_class.return_value)
        self.assertEqual(batch_size, 128)
        print_mock.assert_called_once_with(
            "\nCUDA is available. Using GPU Example GPU for embeddings."
        )

    @patch("src.embeddings.embedding_loader.SentenceTransformer", side_effect=RuntimeError("failed"))
    @patch("src.embeddings.embedding_loader.cuda.is_available", return_value=False)
    def test_model_loading_errors_propagate(self, is_available, model_class) -> None:
        with self.assertRaisesRegex(RuntimeError, "failed"):
            load_sentence_transformer("example-model")


if __name__ == "__main__":
    unittest.main()
