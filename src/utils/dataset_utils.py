"""Format dataset fields and rows as text."""

import pandas as pd

from .text_utils import safe_str


def value_to_text(value: object) -> str:
    return safe_str(value) if pd.api.types.is_scalar(value) else str(value)


def row_to_text(row: pd.Series | dict[str, object]) -> str:
    """
    Convert a dataset row to a formatted text string.
    Args:
        row (pd.Series | dict[str, object]): Dataset row or record fields.
    Returns:
        str: Formatted row text.
    """
    fields = (
        f"{name}: {value_to_text(value)}"
        for name, value in row.items()
    )
    return "\n".join(fields).strip()
