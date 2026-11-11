"""Hash-based identifier utilities."""

import hashlib
from collections.abc import Iterable


def fingerprint_document(data: bytes) -> str:
    """Return the lowercase SHA-256 digest of the original document bytes."""
    return hashlib.sha256(data).hexdigest()


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


def group_positions_by_id(texts: Iterable[str]) -> dict[int, list[int]]:
    """
    Group text positions by their generated IDs.

    Args:
        texts (Iterable[str]): Texts to hash in their original order.

    Returns:
        dict[int, list[int]]: Generated IDs mapped to their text positions.
    """
    groups: dict[int, list[int]] = {}
    for position, text in enumerate(texts):
        groups.setdefault(make_id(text), []).append(position)
    return groups
