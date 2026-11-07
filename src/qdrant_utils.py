import socket


def is_qdrant_running() -> bool:
    """
    Check if Qdrant is running locally on port 6333.
    Returns:
        bool: True if Qdrant is running, False otherwise.
    """
    try:
        socket.create_connection(
            ("localhost", 6333),
            timeout=2,
        )
        return True
    except OSError:
        return False
