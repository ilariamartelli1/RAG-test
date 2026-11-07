import unittest

from src.prompt_utils import message_with_context


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


if __name__ == "__main__":
    unittest.main()
