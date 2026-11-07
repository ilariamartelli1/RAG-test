import pandas as pd
from datasets import load_dataset


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
