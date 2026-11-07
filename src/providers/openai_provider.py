from .http_helper import post_json


class OpenAIProvider:
    """Generate answers using OpenAI's chat completions API."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        max_tokens: int = 1024,
        timeout: float = 120,
    ) -> None:
        """
        Configure the provider.

        Args:
            api_key (str): API key used to authenticate requests.
            model (str): Name of the OpenAI model.
            base_url (str): Base URL of the API.
            max_tokens (int): Maximum number of completion tokens.
            timeout (float): Network timeout in seconds.
        """
        base_url = base_url.rstrip("/")
        if not api_key or not model or not base_url:
            raise ValueError("OpenAI API key, model, and base URL must not be empty")
        if max_tokens <= 0 or timeout <= 0:
            raise ValueError("max_tokens and timeout must be greater than zero")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.timeout = timeout

    def generate(self, messages: list[dict[str, str]], system_prompt: str) -> str:
        """
        Generate an answer using the chat completions API.

        Args:
            messages (list[dict[str, str]]): Messages to send to the model.
            system_prompt (str): Instructions supplied to the model.

        Returns:
            str: Text returned by the model.
        """
        response = post_json(
            f"{self.base_url}/chat/completions",
            {
                "model": self.model,
                "messages": [{"role": "system", "content": system_prompt}, *messages],
                "max_completion_tokens": self.max_tokens,
            },
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout,
        )
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise ValueError("OpenAI response does not contain a completion")
        message = choices[0].get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise ValueError("OpenAI response does not contain message text")
        return message["content"]
