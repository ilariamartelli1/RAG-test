import subprocess
import unittest
from unittest.mock import MagicMock, call, patch
from urllib.error import HTTPError, URLError

from src.storage.qdrant_service import ensure_qdrant, is_qdrant_running, wait_for_qdrant


class QdrantServiceTests(unittest.TestCase):
    @patch("src.storage.qdrant_service.socket.create_connection")
    def test_returns_true_when_connection_succeeds(self, create_connection) -> None:
        self.assertTrue(is_qdrant_running())
        create_connection.assert_called_once_with(("localhost", 6333), timeout=2)

    @patch("src.storage.qdrant_service.socket.create_connection", side_effect=OSError("unavailable"))
    def test_returns_false_when_connection_fails(self, create_connection) -> None:
        self.assertFalse(is_qdrant_running())
        create_connection.assert_called_once_with(("localhost", 6333), timeout=2)

    @patch("src.storage.qdrant_service.subprocess.run")
    @patch("src.storage.qdrant_service.is_qdrant_running", return_value=True)
    @patch("src.storage.qdrant_service.wait_for_qdrant", return_value=True)
    @patch("builtins.print")
    def test_does_not_start_docker_when_already_running(
        self, print_mock, wait, is_running, run
    ) -> None:
        ensure_qdrant("/project/src")

        is_running.assert_called_once_with()
        run.assert_not_called()
        wait.assert_called_once_with()
        self.assertEqual(print_mock.call_args_list, [
            call("Qdrant already running"),
            call("Qdrant is ready"),
        ])

    @patch("src.storage.qdrant_service.subprocess.run")
    @patch("src.storage.qdrant_service.is_qdrant_running", return_value=False)
    @patch("src.storage.qdrant_service.wait_for_qdrant", return_value=True)
    @patch("builtins.print")
    def test_starts_docker_when_not_running(self, print_mock, wait, is_running, run) -> None:
        ensure_qdrant("/project/src")

        is_running.assert_called_once_with()
        run.assert_called_once_with(
            ["docker", "compose", "up", "-d"], cwd="/project/src", check=True
        )
        wait.assert_called_once_with()
        self.assertEqual(print_mock.call_args_list[1].args, ("Qdrant is ready",))

    @patch("src.storage.qdrant_service.wait_for_qdrant")
    @patch("src.storage.qdrant_service.is_qdrant_running", return_value=False)
    @patch("builtins.print")
    @patch("src.storage.qdrant_service.subprocess.run")
    def test_exits_when_docker_fails(self, run, print_mock, is_running, wait) -> None:
        failure = subprocess.CalledProcessError(1, ["docker", "compose", "up", "-d"])
        run.side_effect = failure
        with self.assertRaises(SystemExit) as error:
            ensure_qdrant("/project/src")

        self.assertEqual(error.exception.code, 1)
        is_running.assert_called_once_with()
        wait.assert_not_called()
        print_mock.assert_any_call(f"Failed to start docker compose:\n{failure}")

    @patch("src.storage.qdrant_service.wait_for_qdrant", return_value=False)
    @patch("src.storage.qdrant_service.is_qdrant_running", return_value=True)
    @patch("builtins.print")
    def test_exits_when_qdrant_is_not_ready(self, print_mock, is_running, wait) -> None:
        with self.assertRaises(SystemExit) as error:
            ensure_qdrant("/project/src")

        self.assertEqual(error.exception.code, 1)
        wait.assert_called_once_with()
        print_mock.assert_any_call(
            "Qdrant did not become ready at http://localhost:6333/readyz within 30 seconds."
        )

    @patch("src.storage.qdrant_service.time.sleep")
    @patch("src.storage.qdrant_service.time.monotonic", side_effect=[0, 0, 1, 2])
    @patch("src.storage.qdrant_service.urlopen")
    def test_wait_retries_until_ready(self, urlopen_mock, monotonic, sleep) -> None:
        response = MagicMock()
        response.__enter__.return_value.status = 200
        urlopen_mock.side_effect = [URLError("not ready"), response]

        self.assertTrue(wait_for_qdrant())
        self.assertEqual(urlopen_mock.call_count, 2)
        urlopen_mock.assert_called_with("http://localhost:6333/readyz", timeout=2)
        sleep.assert_called_once_with(1)

    @patch("src.storage.qdrant_service.time.sleep")
    @patch("src.storage.qdrant_service.time.monotonic", side_effect=[0, 0, 0, 1, 2])
    @patch("src.storage.qdrant_service.urlopen")
    def test_wait_times_out_when_endpoint_is_unready(self, urlopen_mock, monotonic, sleep) -> None:
        urlopen_mock.side_effect = HTTPError(
            "http://localhost:6333/readyz", 503, "not ready", {}, None
        )

        self.assertFalse(wait_for_qdrant(timeout=2))
        self.assertEqual(urlopen_mock.call_count, 2)
        self.assertEqual(urlopen_mock.call_args_list[1].kwargs["timeout"], 1)
        sleep.assert_called_once_with(1)


if __name__ == "__main__":
    unittest.main()
