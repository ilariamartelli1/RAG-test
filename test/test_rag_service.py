import unittest
from unittest.mock import MagicMock, patch

from src.chunk_models import ChunkPayload
from src.rag.rag_service import ask


class RagServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.qdrant = MagicMock()
        self.embedder = MagicMock()
        self.llm = MagicMock()
        self.options = {
            "qdrant": self.qdrant,
            "embedder": self.embedder,
            "collection": "documents",
            "embedding_batch_size": 16,
            "llm": self.llm,
            "system_prompt": "Instructions",
        }

    @patch("src.rag.rag_service.retrieve_chunks")
    @patch("builtins.print")
    def test_builds_prompt_and_returns_model_answer(self, print_mock, retrieve) -> None:
        retrieve.return_value = [(ChunkPayload(text="Context", metadata={}), 0.9)]
        self.llm.generate.return_value = " Answer "

        result = ask("Question", **self.options)

        self.assertEqual(result, "Answer")
        retrieve.assert_called_once_with(self.qdrant, self.embedder, "documents", "Question", 16)
        self.llm.generate.assert_called_once_with(
            system_prompt="Instructions",
            messages=[{
                "role": "user",
                "content": (
                    "Here is the relevant context from the document:\n\n"
                    "Context\n\n---\n\nUser question: Question"
                ),
            }],
        )

    @patch("src.rag.rag_service.retrieve_chunks", return_value=[])
    @patch("builtins.print")
    def test_accepts_provider_without_anthropic_client_interface(self, print_mock, retrieve) -> None:
        class LocalProvider:
            def generate(self, messages: list[dict[str, str]], system_prompt: str) -> str:
                return "Local answer"

        self.options["llm"] = LocalProvider()
        self.assertEqual(ask("Question", **self.options), "Local answer")

    @patch("src.rag.rag_service.time.sleep")
    @patch("src.rag.rag_service.retrieve_chunks", return_value=[])
    @patch("builtins.print")
    def test_retries_model_failures(self, print_mock, retrieve, sleep) -> None:
        self.llm.generate.side_effect = [
            RuntimeError("unavailable"),
            "Answer",
        ]

        self.assertEqual(ask("Question", **self.options), "Answer")

        self.assertEqual(self.llm.generate.call_count, 2)
        sleep.assert_called_once_with(1)

    @patch("src.rag.rag_service.time.sleep")
    @patch("src.rag.rag_service.retrieve_chunks", return_value=[])
    @patch("builtins.print")
    def test_raises_after_last_attempt(self, print_mock, retrieve, sleep) -> None:
        self.llm.generate.side_effect = RuntimeError("unavailable")

        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            ask("Question", retries=2, **self.options)

        self.assertEqual(self.llm.generate.call_count, 2)
        sleep.assert_called_once_with(1)


if __name__ == "__main__":
    unittest.main()
