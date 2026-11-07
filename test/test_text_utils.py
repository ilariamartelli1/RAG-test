import unittest

import pandas as pd

from src.text_utils import safe_str


class SafeStrTests(unittest.TestCase):
    def test_missing_values_become_empty_strings(self) -> None:
        for value in (None, float("nan"), pd.NA, pd.NaT):
            with self.subTest(value=value):
                self.assertEqual(safe_str(value), "")

    def test_text_is_preserved(self) -> None:
        self.assertEqual(safe_str("sample text"), "sample text")
        self.assertEqual(safe_str(""), "")

    def test_other_values_are_converted_to_strings(self) -> None:
        self.assertEqual(safe_str(42), "42")
        self.assertEqual(safe_str(False), "False")


if __name__ == "__main__":
    unittest.main()
