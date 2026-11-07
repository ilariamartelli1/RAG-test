import unittest
from unittest.mock import patch

import pandas as pd
from pandas.testing import assert_frame_equal

from src.processing.dataset_loader import load_dataset_dataframe


class DatasetLoaderTests(unittest.TestCase):
    @patch("src.processing.dataset_loader.load_dataset")
    def test_loads_specified_split_and_drops_columns(self, load_dataset_mock) -> None:
        load_dataset_mock.return_value.to_pandas.return_value = pd.DataFrame({
            "title": ["Example"],
            "unused": [1],
        })

        result = load_dataset_dataframe("example/dataset", "validation", ["unused"])

        load_dataset_mock.assert_called_once_with("example/dataset", split="validation")
        assert_frame_equal(result, pd.DataFrame({"title": ["Example"]}))

    @patch("src.processing.dataset_loader.load_dataset")
    def test_keeps_all_columns_by_default(self, load_dataset_mock) -> None:
        dataframe = pd.DataFrame({"first": [1], "second": [2]})
        load_dataset_mock.return_value.to_pandas.return_value = dataframe

        result = load_dataset_dataframe("example/dataset", "train")

        assert_frame_equal(result, dataframe)

    @patch("src.processing.dataset_loader.load_dataset")
    def test_missing_column_raises_key_error(self, load_dataset_mock) -> None:
        load_dataset_mock.return_value.to_pandas.return_value = pd.DataFrame({"field": [1]})

        with self.assertRaises(KeyError):
            load_dataset_dataframe("example/dataset", "train", ["missing"])

    @patch("src.processing.dataset_loader.load_dataset", side_effect=RuntimeError("unavailable"))
    def test_loading_errors_propagate(self, load_dataset_mock) -> None:
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            load_dataset_dataframe("example/dataset", "train")


if __name__ == "__main__":
    unittest.main()
