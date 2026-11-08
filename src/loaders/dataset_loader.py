import pandas as pd
from datasets import load_dataset

if "." in __package__:
    from ..processing.record_models import SourceRecord
else:
    from processing.record_models import SourceRecord


def load_dataset_dataframe(
    dataset_name: str,
    split: str,
    drop_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load a dataset split as a pandas DataFrame.

    Args:
        dataset_name (str): Name or path passed to the dataset loader.
        split (str): Dataset split to load.
        drop_columns (list[str] | None): Columns to remove, if any.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    dataset = load_dataset(dataset_name, split=split)
    dataframe = dataset.to_pandas()
    if drop_columns is not None:
        dataframe = dataframe.drop(columns=drop_columns)
    return dataframe


def dataframe_to_records(dataframe: pd.DataFrame, source: str) -> list[SourceRecord]:
    """
    Convert dataset rows to source records without modifying the DataFrame.

    Args:
        dataframe (pd.DataFrame): Dataset rows and their field values.
        source (str): Dataset source recorded in metadata.

    Returns:
        list[SourceRecord]: One record per row, in DataFrame order.

    Raises:
        ValueError: If column names are non-string or duplicated, or row
            identifiers are not unique when converted to strings.
    """
    if not dataframe.columns.is_unique or not all(
        isinstance(column, str) for column in dataframe.columns
    ):
        raise ValueError("Dataset column names must be unique strings")

    row_ids = [str(index) for index in dataframe.index]
    if len(set(row_ids)) != len(row_ids):
        raise ValueError("Dataset row identifiers must be unique as strings")

    return [
        SourceRecord(
            content=dict(zip(dataframe.columns, row[1:], strict=True)),
            metadata={"source": source, "row_id": row_id},
        )
        for row_id, row in zip(
            row_ids, dataframe.itertuples(index=True, name=None), strict=True
        )
    ]


def load_dataset_records(
    dataset_name: str,
    split: str,
    drop_columns: list[str] | None = None,
) -> list[SourceRecord]:
    """
    Load a dataset split and convert its rows to source records.

    Args:
        dataset_name (str): Name or path passed to the dataset loader.
        split (str): Dataset split to load.
        drop_columns (list[str] | None): Columns to remove, if any.

    Returns:
        list[SourceRecord]: Dataset rows with source and row ID metadata.
    """
    dataframe = load_dataset_dataframe(dataset_name, split, drop_columns)
    return dataframe_to_records(dataframe, dataset_name)
