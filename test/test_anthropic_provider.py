"""Tests for the Anthropic provider adapter."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.providers.anthropic_provider import AnthropicProvider


class AnthropicProviderTests(unittest.TestCase):
    def test_generate_passes_model_settings_and_returns_text(self) -> None:
        client = MagicMock()
        client.messages.create.return_value = SimpleNamespace(
            content=[SimpleNamespace(text=" Answer ")]
        )
        provider = AnthropicProvider(client, "model", max_tokens=256)
        messages = [{"role": "user", "content": "Question"}]

        self.assertEqual(provider.generate(messages, "Instructions"), " Answer ")
        client.messages.create.assert_called_once_with(
            model="model",
            max_tokens=256,
            system="Instructions",
            messages=messages,
        )

    def test_generate_uses_default_max_tokens(self) -> None:
        client = MagicMock()
        client.messages.create.return_value = SimpleNamespace(
            content=[SimpleNamespace(text="Answer")]
        )

        AnthropicProvider(client, "model").generate([], "Instructions")

        self.assertEqual(client.messages.create.call_args.kwargs["max_tokens"], 1024)

    def test_generate_propagates_client_errors(self) -> None:
        client = MagicMock()
        client.messages.create.side_effect = RuntimeError("unavailable")

        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            AnthropicProvider(client, "model").generate([], "Instructions")


if __name__ == "__main__":
    unittest.main()
