import unittest
from unittest.mock import patch

import pandas as pd
from pandas.testing import assert_frame_equal

from src.processing.dataset_loader import dataframe_to_records, load_dataset_dataframe


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

    def test_converts_rows_to_records_with_source_and_row_ids(self) -> None:
        dataframe = pd.DataFrame(
            {"title": ["First", "Second"], "count": [1, 2], "tags": [["a"], []]},
            index=[10, 20],
        )
        original = dataframe.copy(deep=True)

        records = dataframe_to_records(dataframe, "example/dataset")

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].content, {"title": "First", "count": 1, "tags": ["a"]})
        self.assertEqual(records[0].metadata, {"source": "example/dataset", "row_id": "10"})
        self.assertEqual(records[1].content, {"title": "Second", "count": 2, "tags": []})
        self.assertEqual(records[1].metadata, {"source": "example/dataset", "row_id": "20"})
        assert_frame_equal(dataframe, original)

    def test_converts_empty_dataframe(self) -> None:
        self.assertEqual(dataframe_to_records(pd.DataFrame(columns=["title"]), "dataset"), [])

    def test_converts_rows_without_columns(self) -> None:
        records = dataframe_to_records(pd.DataFrame(index=[1, 2]), "dataset")

        self.assertEqual([record.content for record in records], [{}, {}])
        self.assertEqual([record.metadata["row_id"] for record in records], ["1", "2"])

    def test_rejects_columns_that_cannot_form_field_mapping(self) -> None:
        for columns in (["field", "field"], ["field", 1]):
            dataframe = pd.DataFrame([[1, 2]], columns=columns)
            with self.subTest(columns=columns), self.assertRaisesRegex(ValueError, "column names"):
                dataframe_to_records(dataframe, "dataset")

    def test_rejects_row_ids_that_collide_as_strings(self) -> None:
        dataframe = pd.DataFrame({"field": [1, 2]}, index=pd.Index([1, "1"], dtype=object))

        with self.assertRaisesRegex(ValueError, "row identifiers"):
            dataframe_to_records(dataframe, "dataset")


if __name__ == "__main__":
    unittest.main()
