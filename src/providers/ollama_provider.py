"""Provide language model generation through an Ollama server."""

from .http_helper import post_json


class OllamaProvider:
    """Generate answers using a local or remote Ollama server."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        max_tokens: int = 1024,
        timeout: float = 120,
    ) -> None:
        """
        Configure the provider.

        Args:
            model (str): Name of a model installed on the Ollama server.
            base_url (str): Base URL of the Ollama server.
            max_tokens (int): Maximum number of generated tokens.
            timeout (float): Network timeout in seconds.
        """
        base_url = base_url.rstrip("/")
        if not model or not base_url:
            raise ValueError("Ollama model and base URL must not be empty")
        if max_tokens <= 0 or timeout <= 0:
            raise ValueError("max_tokens and timeout must be greater than zero")
        self.model = model
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.timeout = timeout

    def generate(self, messages: list[dict[str, str]], system_prompt: str) -> str:
        """
        Generate an answer using Ollama's chat API.

        Args:
            messages (list[dict[str, str]]): Messages to send to the model.
            system_prompt (str): Instructions supplied to the model.

        Returns:
            str: Text returned by the model.
        """
        response = post_json(
            f"{self.base_url}/api/chat",
            {
                "model": self.model,
                "messages": [{"role": "system", "content": system_prompt}, *messages],
                "stream": False,
                "options": {"num_predict": self.max_tokens},
            },
            headers={},
            timeout=self.timeout,
        )
        message = response.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise ValueError("Ollama response does not contain message text")
        return message["content"]
