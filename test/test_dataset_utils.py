import unittest

import pandas as pd

from src.dataset_utils import row_to_text


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


if __name__ == "__main__":
    unittest.main()
