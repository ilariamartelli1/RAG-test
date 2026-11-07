import unittest
from unittest.mock import patch

from src.network_utils import is_online


class NetworkUtilsTests(unittest.TestCase):
    @patch("src.network_utils.socket.create_connection")
    def test_returns_normally_when_connection_succeeds(self, create_connection) -> None:
        self.assertIsNone(is_online())
        create_connection.assert_called_once_with(("8.8.8.8", 53), timeout=5)

    @patch("builtins.print")
    @patch("src.network_utils.socket.create_connection", side_effect=OSError("unavailable"))
    def test_exits_with_error_when_connection_fails(self, create_connection, print_mock) -> None:
        with self.assertRaises(SystemExit) as error:
            is_online()

        self.assertEqual(error.exception.code, 1)
        create_connection.assert_called_once_with(("8.8.8.8", 53), timeout=5)
        print_mock.assert_called_once_with(
            "Could not connect to 8.8.8.8 on port 53:\nunavailable"
        )


if __name__ == "__main__":
    unittest.main()
