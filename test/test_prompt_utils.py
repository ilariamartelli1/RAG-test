"""Tests for retrieval-augmented prompt construction."""

import unittest

from src.rag.prompt_utils import build_rag_prompt, message_with_context


class PromptUtilsTests(unittest.TestCase):
    def test_formats_question_and_context(self) -> None:
        self.assertEqual(
            message_with_context("What can I cook?", "Tomato soup"),
            "Here is the relevant context from the document:\n\n"
            "Tomato soup\n\n---\n\n"
            "User question: What can I cook?",
        )

    def test_formats_empty_question_and_context(self) -> None:
        self.assertEqual(
            message_with_context("", ""),
            "Here is the relevant context from the document:\n\n"
            "\n\n---\n\n"
            "User question: ",
        )

    def test_builds_prompt_from_generic_chunks(self) -> None:
        self.assertEqual(
            build_rag_prompt("What is relevant?", ["First chunk", "Second chunk"]),
            [{
                "role": "user",
                "content": (
                    "Here is the relevant context from the document:\n\n"
                    "First chunk\n\n---\n\nSecond chunk"
                    "\n\n---\n\nUser question: What is relevant?"
                ),
            }],
        )

    def test_builds_prompt_without_chunks(self) -> None:
        self.assertEqual(
            build_rag_prompt("Question", []),
            [{
                "role": "user",
                "content": (
                    "Here is the relevant context from the document:\n\n"
                    "\n\n---\n\nUser question: Question"
                ),
            }],
        )


if __name__ == "__main__":
    unittest.main()
