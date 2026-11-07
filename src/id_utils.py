"""Hash-based identifier utilities."""

import hashlib


def make_id(text: str) -> int:
    """
    Generate a numeric ID from MD5 hash.
    Args:
        text (str): Input text.
    Returns:
        int: 64-bit integer ID.
    """
    hash_hex = hashlib.md5(text.encode()).hexdigest()
    # Convert first 16 hex characters to a 64-bit integer
    return int(hash_hex[:16], 16)
