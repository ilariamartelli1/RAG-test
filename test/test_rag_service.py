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

    @patch("src.rag.rag_service.retrieve_chunks")
    @patch("builtins.print")
    def test_accepts_provider_without_anthropic_client_interface(self, print_mock, retrieve) -> None:
        retrieve.return_value = [(ChunkPayload(text="Context", metadata={}), 0.9)]

        class LocalProvider:
            def generate(self, messages: list[dict[str, str]], system_prompt: str) -> str:
                return "Local answer"

        self.options["llm"] = LocalProvider()
        self.assertEqual(ask("Question", **self.options), "Local answer")

    @patch("src.rag.rag_service.time.sleep")
    @patch("src.rag.rag_service.retrieve_chunks")
    @patch("builtins.print")
    def test_retries_model_failures(self, print_mock, retrieve, sleep) -> None:
        retrieve.return_value = [(ChunkPayload(text="Context", metadata={}), 0.9)]
        self.llm.generate.side_effect = [
            RuntimeError("unavailable"),
            "Answer",
        ]

        self.assertEqual(ask("Question", **self.options), "Answer")

        self.assertEqual(self.llm.generate.call_count, 2)
        sleep.assert_called_once_with(1)

    @patch("src.rag.rag_service.time.sleep")
    @patch("src.rag.rag_service.retrieve_chunks")
    @patch("builtins.print")
    def test_raises_after_last_attempt(self, print_mock, retrieve, sleep) -> None:
        retrieve.return_value = [(ChunkPayload(text="Context", metadata={}), 0.9)]
        self.llm.generate.side_effect = RuntimeError("unavailable")

        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            ask("Question", retries=2, **self.options)

        self.assertEqual(self.llm.generate.call_count, 2)
        sleep.assert_called_once_with(1)

    @patch("src.rag.rag_service.build_rag_prompt")
    @patch("src.rag.rag_service.retrieve_chunks", return_value=[])
    @patch("builtins.print")
    def test_no_results_returns_message_without_calling_model(
        self, print_mock, retrieve, build_prompt
    ) -> None:
        answer = ask("Question", **self.options)

        self.assertEqual(
            answer,
            "No relevant information was retrieved for your question. "
            "If the collection contains relevant information, try lowering "
            "the retrieval score threshold.",
        )
        print_mock.assert_any_call(f"Answer:\n{answer}")
        build_prompt.assert_not_called()
        self.llm.generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
