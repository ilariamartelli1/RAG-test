import unittest

import pandas as pd

from src.dataset_utils import chunk_dataset_row, row_to_text


class DatasetUtilsTests(unittest.TestCase):
    def test_formats_fields(self) -> None:
        row = pd.Series({
            "Title": "Soup",
            "Ingredients": "Tomato",
            "Instructions": "Cook",
        })

        self.assertEqual(
            row_to_text(row),
            "Title: Soup\nIngredients: Tomato\nInstructions: Cook",
        )

    def test_formats_missing_and_non_string_values(self) -> None:
        row = pd.Series(
            {"Title": 42, "Ingredients": None, "Instructions": float("nan")},
            dtype=object,
        )

        self.assertEqual(
            row_to_text(row),
            "Title: 42\nIngredients: \nInstructions:",
        )

    def test_formats_different_fields_in_order(self) -> None:
        row = pd.Series({"author": "Ada", "year": 2024, "tags": ["science", "history"]})

        self.assertEqual(
            row_to_text(row),
            "author: Ada\nyear: 2024\ntags: ['science', 'history']",
        )

    def test_empty_row_returns_empty_text(self) -> None:
        self.assertEqual(row_to_text(pd.Series(dtype=object)), "")

    def test_chunk_keeps_a_row_together_when_it_fits(self) -> None:
        row = pd.Series({"name": "Soup", "ingredients": "Tomato"})
        expected = "Record: row-1\nname: Soup\ningredients: Tomato"

        chunks = chunk_dataset_row(row, "row-1", max_chunk_chars=len(expected))

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, expected)
        self.assertEqual(chunks[0].row_id, "row-1")
        self.assertEqual(chunks[0].chunk_index, 0)
        self.assertIsNone(chunks[0].field_name)

    def test_chunk_splits_long_rows_by_field_and_preserves_values(self) -> None:
        row = pd.Series({"ingredients": "red tomato and onion", "steps": "cook slowly"})
        chunks = chunk_dataset_row(row, "1", max_chunk_chars=32)

        self.assertTrue(all(len(chunk.text) <= 32 for chunk in chunks))
        self.assertTrue(all(chunk.row_id == "1" for chunk in chunks))
        self.assertEqual([chunk.chunk_index for chunk in chunks], list(range(len(chunks))))
        self.assertEqual({chunk.field_name for chunk in chunks}, {"ingredients", "steps"})
        for field in ("ingredients", "steps"):
            prefix = f"Record: 1\n{field}: "
            values = [chunk.text[len(prefix):] for chunk in chunks if chunk.field_name == field]
            self.assertEqual("".join(values), row[field])

    def test_chunk_splits_long_unbroken_field(self) -> None:
        chunks = chunk_dataset_row(
            pd.Series({"data": "abcdefghij"}), "1", max_chunk_chars=18
        )

        self.assertEqual([chunk.text for chunk in chunks], [
            "Record: 1\ndata: ab",
            "Record: 1\ndata: cd",
            "Record: 1\ndata: ef",
            "Record: 1\ndata: gh",
            "Record: 1\ndata: ij",
        ])

    def test_chunk_rejects_invalid_size_and_row_id(self) -> None:
        row = pd.Series({"field": "value"})
        for row_id, max_chunk_chars in (("1", 0), ("", 100), ("1", 10)):
            with self.subTest(row_id=row_id, max_chunk_chars=max_chunk_chars):
                with self.assertRaises(ValueError):
                    chunk_dataset_row(row, row_id, max_chunk_chars)

    def test_chunk_empty_row(self) -> None:
        self.assertEqual(chunk_dataset_row(pd.Series(dtype=object), "1"), [])


if __name__ == "__main__":
    unittest.main()
