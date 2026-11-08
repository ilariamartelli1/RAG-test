"""Start Qdrant when needed and wait for its readiness endpoint."""

import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen


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


def wait_for_qdrant(timeout: float = 30) -> bool:
    """Wait for Qdrant's readiness endpoint until the timeout expires."""
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        try:
            with urlopen("http://localhost:6333/readyz", timeout=min(2, remaining)) as response:
                if response.status == 200:
                    return True
        except (URLError, OSError):
            pass

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(1, remaining))


def ensure_qdrant(project_root: str) -> None:
    """
    Ensure Qdrant is running, start it with Docker Compose if not.
    """
    if is_qdrant_running():
        print("Qdrant already running")
    else:
        print("Starting Qdrant...")

        try:
            subprocess.run(
                ["docker", "compose", "up", "-d"],
                cwd=project_root,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"Failed to start docker compose:\n{e}")
            sys.exit(1)

    if not wait_for_qdrant():
        print("Qdrant did not become ready at http://localhost:6333/readyz within 30 seconds.")
        sys.exit(1)
    print("Qdrant is ready")
