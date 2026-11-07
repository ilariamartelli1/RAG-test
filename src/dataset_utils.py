import pandas as pd

if __package__:
    from .text_utils import safe_str
else:
    from text_utils import safe_str


def row_to_text(row: pd.Series) -> str:
    """
    Convert a dataset row to a formatted text string.
    Args:
        row (pd.Series): Dataset row.
    Returns:
        str: Formatted row text.
    """
    fields = (
        f"{name}: {safe_str(value) if pd.api.types.is_scalar(value) else str(value)}"
        for name, value in row.items()
    )
    return "\n".join(fields).strip()
