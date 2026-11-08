"""Tests for the Ollama provider adapter."""

import unittest
from unittest.mock import patch
from urllib.error import URLError

from src.providers.ollama_provider import OllamaProvider


class OllamaProviderTests(unittest.TestCase):
    @patch("src.providers.ollama_provider.post_json")
    def test_generate_sends_chat_request_and_returns_text(self, post_json) -> None:
        post_json.return_value = {"message": {"content": "Answer"}}
        provider = OllamaProvider("local-model", "http://localhost:11434/", 256, 30)
        messages = [{"role": "user", "content": "Question"}]

        self.assertEqual(provider.generate(messages, "Instructions"), "Answer")
        post_json.assert_called_once_with(
            "http://localhost:11434/api/chat",
            {
                "model": "local-model",
                "messages": [
                    {"role": "system", "content": "Instructions"},
                    {"role": "user", "content": "Question"},
                ],
                "stream": False,
                "options": {"num_predict": 256},
            },
            headers={},
            timeout=30,
        )
        self.assertEqual(messages, [{"role": "user", "content": "Question"}])

    @patch("src.providers.ollama_provider.post_json", return_value={"message": {}})
    def test_rejects_response_without_message_text(self, post_json) -> None:
        with self.assertRaisesRegex(ValueError, "Ollama response"):
            OllamaProvider("local-model").generate([], "Instructions")

    @patch("src.providers.ollama_provider.post_json", side_effect=URLError("unavailable"))
    def test_propagates_server_errors(self, post_json) -> None:
        with self.assertRaises(URLError):
            OllamaProvider("local-model").generate([], "Instructions")

    def test_rejects_invalid_settings(self) -> None:
        for kwargs in ({"model": ""}, {"model": "m", "max_tokens": 0},
                       {"model": "m", "timeout": 0}, {"model": "m", "base_url": "/"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                OllamaProvider(**kwargs)


if __name__ == "__main__":
    unittest.main()
