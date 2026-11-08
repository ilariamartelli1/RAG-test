import unittest

import pandas as pd

from src.processing.dataset_utils import row_to_text


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

    def test_formats_record_fields_like_a_series(self) -> None:
        fields: dict[str, object] = {
            "author": "Ada",
            "year": 2024,
            "missing": None,
            "tags": ["science", "history"],
        }

        self.assertEqual(
            row_to_text(fields),
            "author: Ada\nyear: 2024\nmissing: \ntags: ['science', 'history']",
        )
        self.assertEqual(row_to_text(fields), row_to_text(pd.Series(fields, dtype=object)))

    def test_empty_record_fields_return_empty_text(self) -> None:
        self.assertEqual(row_to_text({}), "")


if __name__ == "__main__":
    unittest.main()
