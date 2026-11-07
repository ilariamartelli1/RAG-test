import unittest

from src.id_utils import make_id


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


if __name__ == "__main__":
    unittest.main()
