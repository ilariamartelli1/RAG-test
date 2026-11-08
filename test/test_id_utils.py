"""Tests for hash identifiers and duplicate grouping."""

import unittest
from unittest.mock import patch

from src.utils.id_utils import group_positions_by_id, make_id


class IdUtilsTests(unittest.TestCase):
    def test_uses_first_64_bits_of_md5_digest(self) -> None:
        self.assertEqual(make_id("hello"), int("5d41402abc4b2a76", 16))

    def test_returns_unsigned_64_bit_integer(self) -> None:
        result = make_id("sample text")

        self.assertIsInstance(result, int)
        self.assertGreaterEqual(result, 0)
        self.assertLess(result, 2**64)

    def test_is_stable_for_the_same_input(self) -> None:
        self.assertEqual(make_id("sample text"), make_id("sample text"))

    def test_different_inputs_produce_different_ids(self) -> None:
        self.assertNotEqual(make_id("first sample"), make_id("second sample"))

    def test_groups_duplicate_text_positions(self) -> None:
        texts = ["first", "second", "first", "second", "first"]

        self.assertEqual(group_positions_by_id(texts), {
            make_id("first"): [0, 2, 4],
            make_id("second"): [1, 3],
        })

    def test_accepts_empty_iterable(self) -> None:
        self.assertEqual(group_positions_by_id([]), {})

    def test_accepts_generator(self) -> None:
        self.assertEqual(
            group_positions_by_id(text for text in ("first", "first")),
            {make_id("first"): [0, 1]},
        )

    @patch("src.utils.id_utils.make_id", return_value=123)
    def test_groups_hash_collisions(self, make_id_mock) -> None:
        self.assertEqual(group_positions_by_id(["first", "second"]), {123: [0, 1]})


if __name__ == "__main__":
    unittest.main()
