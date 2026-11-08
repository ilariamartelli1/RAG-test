"""Check network connectivity before running the pipeline."""

import socket
import sys


def is_online() -> None:
    """
    Check if the machine is online by connecting to a public DNS server.
    Exits the program if offline.
    """
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=5)
    except OSError as e:
        print(f"Could not connect to 8.8.8.8 on port 53:\n{e}")
        sys.exit(1)
