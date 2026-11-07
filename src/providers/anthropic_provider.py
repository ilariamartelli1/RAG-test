from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anthropic import Anthropic


class AnthropicProvider:
    """Generate answers using an Anthropic client."""

    def __init__(self, client: Anthropic, model: str, max_tokens: int = 1024) -> None:
        """
        Configure the provider.

        Args:
            client (Anthropic): Client used to call Anthropic.
            model (str): Name of the Anthropic model.
            max_tokens (int): Maximum number of tokens in the response.
        """
        self.client = client
        self.model = model
        self.max_tokens = max_tokens

    def generate(self, messages: list[dict[str, str]], system_prompt: str) -> str:
        """
        Generate an answer from the supplied messages.

        Args:
            messages (list[dict[str, str]]): Messages to send to the model.
            system_prompt (str): Instructions supplied to the model.

        Returns:
            str: Text returned by the model.
        """
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_prompt,
            messages=messages,
        )
        return response.content[0].text
