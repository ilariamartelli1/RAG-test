import unittest

from pydantic import ValidationError

from src.models.chunk_models import ChunkPayload


class ChunkModelsTests(unittest.TestCase):
    def test_payload_contains_model_fields(self) -> None:
        payload = ChunkPayload(
            text="Example text",
            metadata={"source": "example.txt", "page": 1},
        ).model_dump()

        self.assertEqual(payload, {
            "text": "Example text",
            "metadata": {"source": "example.txt", "page": 1},
        })

    def test_missing_field_fails_validation(self) -> None:
        with self.assertRaises(ValidationError):
            ChunkPayload.model_validate({"text": "Example text"})

    def test_metadata_must_be_json_compatible(self) -> None:
        with self.assertRaises(ValidationError):
            ChunkPayload(text="Example text", metadata={"source": object()})


if __name__ == "__main__":
    unittest.main()
