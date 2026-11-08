"""Tests for the provider JSON HTTP helper."""

import io
import json
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from unittest.mock import patch
from urllib.error import URLError

from src.providers.http_helper import post_json


class HttpHelperTests(unittest.TestCase):
    def test_posts_to_local_http_server(self) -> None:
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                body = self.rfile.read(int(self.headers["Content-Length"]))
                requests.append((self.path, json.loads(body)))
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"answer": "ok"}')

            def log_message(self, format, *args) -> None:
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        server.timeout = 2
        thread = Thread(target=server.handle_request)
        thread.start()
        try:
            response = post_json(
                f"http://127.0.0.1:{server.server_port}/chat", {"model": "example"}, {}, 2
            )
        finally:
            thread.join(timeout=3)
            server.server_close()

        self.assertEqual(response, {"answer": "ok"})
        self.assertEqual(requests, [("/chat", {"model": "example"})])
        self.assertFalse(thread.is_alive())

    @patch("src.providers.http_helper.urlopen")
    def test_posts_json_and_returns_object(self, urlopen_mock) -> None:
        urlopen_mock.return_value.__enter__.return_value = io.BytesIO(b'{"answer": "ok"}')

        result = post_json(
            "http://localhost:1234/chat",
            {"model": "example"},
            {"Authorization": "Bearer test-key"},
            timeout=10,
        )

        self.assertEqual(result, {"answer": "ok"})
        request = urlopen_mock.call_args.args[0]
        self.assertEqual(request.full_url, "http://localhost:1234/chat")
        self.assertEqual(request.method, "POST")
        self.assertEqual(json.loads(request.data), {"model": "example"})
        self.assertEqual(request.get_header("Content-type"), "application/json")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
        self.assertEqual(urlopen_mock.call_args.kwargs["timeout"], 10)

    @patch("src.providers.http_helper.urlopen")
    def test_rejects_non_object_response(self, urlopen_mock) -> None:
        urlopen_mock.return_value.__enter__.return_value = io.BytesIO(b"[]")

        with self.assertRaisesRegex(ValueError, "JSON object"):
            post_json("http://localhost:1234", {}, {}, 10)

    @patch("src.providers.http_helper.urlopen", side_effect=URLError("unavailable"))
    def test_propagates_network_errors(self, urlopen_mock) -> None:
        with self.assertRaises(URLError):
            post_json("http://localhost:1234", {}, {}, 10)


if __name__ == "__main__":
    unittest.main()
