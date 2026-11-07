import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.chunk_models import ChunkPayload
from src.rag_service import ask


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
            "llm_model": "model",
            "system_prompt": "Instructions",
        }

    @patch("src.rag_service.retrieve_chunks")
    def test_builds_prompt_and_returns_model_answer(self, retrieve) -> None:
        retrieve.return_value = [(ChunkPayload(text="Context", metadata={}), 0.9)]
        self.llm.messages.create.return_value = SimpleNamespace(
            content=[SimpleNamespace(text=" Answer ")]
        )

        with patch("builtins.print"):
            result = ask("Question", **self.options)

        self.assertEqual(result, "Answer")
        retrieve.assert_called_once_with(self.qdrant, self.embedder, "documents", "Question", 16)
        self.llm.messages.create.assert_called_once_with(
            model="model",
            max_tokens=1024,
            system="Instructions",
            messages=[{
                "role": "user",
                "content": (
                    "Here is the relevant context from the document:\n\n"
                    "Context\n\n---\n\nUser question: Question"
                ),
            }],
        )

    @patch("src.rag_service.time.sleep")
    @patch("src.rag_service.retrieve_chunks", return_value=[])
    def test_retries_model_failures(self, retrieve, sleep) -> None:
        self.llm.messages.create.side_effect = [
            RuntimeError("unavailable"),
            SimpleNamespace(content=[SimpleNamespace(text="Answer")]),
        ]

        with patch("builtins.print"):
            self.assertEqual(ask("Question", **self.options), "Answer")

        self.assertEqual(self.llm.messages.create.call_count, 2)
        sleep.assert_called_once_with(1)

    @patch("src.rag_service.time.sleep")
    @patch("src.rag_service.retrieve_chunks", return_value=[])
    def test_raises_after_last_attempt(self, retrieve, sleep) -> None:
        self.llm.messages.create.side_effect = RuntimeError("unavailable")

        with patch("builtins.print"), self.assertRaisesRegex(RuntimeError, "unavailable"):
            ask("Question", retries=2, **self.options)

        self.assertEqual(self.llm.messages.create.call_count, 2)
        sleep.assert_called_once_with(1)


if __name__ == "__main__":
    unittest.main()
