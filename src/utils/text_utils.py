"""Convert scalar values to text while handling missing values."""

from typing import Any

import pandas as pd


def safe_str(val: Any) -> str:
    """
    Convert a value to string, handling NaN/None.
    Args:
        val: Value to convert.
    Returns:
        str: String representation or empty string.
    """
    if pd.isna(val):
        return ""
    return str(val)
