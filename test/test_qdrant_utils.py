import unittest
from unittest.mock import patch

from src.qdrant_utils import is_qdrant_running


class QdrantUtilsTests(unittest.TestCase):
    @patch("src.qdrant_utils.socket.create_connection")
    def test_returns_true_when_connection_succeeds(self, create_connection) -> None:
        self.assertTrue(is_qdrant_running())
        create_connection.assert_called_once_with(("localhost", 6333), timeout=2)

    @patch("src.qdrant_utils.socket.create_connection", side_effect=OSError("unavailable"))
    def test_returns_false_when_connection_fails(self, create_connection) -> None:
        self.assertFalse(is_qdrant_running())
        create_connection.assert_called_once_with(("localhost", 6333), timeout=2)


if __name__ == "__main__":
    unittest.main()
