"""Send JSON POST requests for HTTP-based language model providers."""

import json
from urllib.request import Request, urlopen


def post_json(
    url: str, payload: dict[str, object], headers: dict[str, str], timeout: float
) -> dict[str, object]:
    """
    Send a JSON POST request and decode its JSON object response.

    Args:
        url (str): Endpoint URL.
        payload (dict[str, object]): Request body.
        headers (dict[str, str]): Additional HTTP headers.
        timeout (float): Network timeout in seconds.

    Returns:
        dict[str, object]: Decoded response object.
    """
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    if not isinstance(result, dict):
        raise ValueError("Provider response must be a JSON object")
    return result
