import unittest
from unittest.mock import patch
from urllib.error import URLError

from src.providers.openai_provider import OpenAIProvider


class OpenAIProviderTests(unittest.TestCase):
    @patch("src.providers.openai_provider.post_json")
    def test_generate_sends_chat_request_and_returns_text(self, post_json) -> None:
        post_json.return_value = {"choices": [{"message": {"content": "Answer"}}]}
        provider = OpenAIProvider("test-key", "example-model", "https://example.test/v1/", 256, 30)
        messages = [{"role": "user", "content": "Question"}]

        self.assertEqual(provider.generate(messages, "Instructions"), "Answer")
        post_json.assert_called_once_with(
            "https://example.test/v1/chat/completions",
            {
                "model": "example-model",
                "messages": [
                    {"role": "system", "content": "Instructions"},
                    {"role": "user", "content": "Question"},
                ],
                "max_completion_tokens": 256,
            },
            headers={"Authorization": "Bearer test-key"},
            timeout=30,
        )
        self.assertEqual(messages, [{"role": "user", "content": "Question"}])

    @patch("src.providers.openai_provider.post_json")
    def test_rejects_response_without_message_text(self, post_json) -> None:
        for response in ({}, {"choices": []}, {"choices": [{"message": {"content": None}}]}):
            post_json.return_value = response
            with self.subTest(response=response), self.assertRaisesRegex(ValueError, "OpenAI response"):
                OpenAIProvider("test-key", "model").generate([], "Instructions")

    @patch("src.providers.openai_provider.post_json", side_effect=URLError("unavailable"))
    def test_propagates_server_errors(self, post_json) -> None:
        with self.assertRaises(URLError):
            OpenAIProvider("test-key", "model").generate([], "Instructions")

    def test_rejects_invalid_settings(self) -> None:
        for kwargs in (
            {"api_key": "", "model": "m"},
            {"api_key": "test-key", "model": ""},
            {"api_key": "test-key", "model": "m", "max_tokens": 0},
            {"api_key": "test-key", "model": "m", "timeout": 0},
            {"api_key": "test-key", "model": "m", "base_url": "/"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                OpenAIProvider(**kwargs)


if __name__ == "__main__":
    unittest.main()
