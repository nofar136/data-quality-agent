"""Small shared helper functions used across the data quality agent."""

from __future__ import annotations


def human_readable_size(num_bytes: float) -> str:
    """Convert a byte count into a human-readable string.

    Args:
        num_bytes: Size in bytes.

    Returns:
        A string such as "1.5 KB" or "3.2 MB".

    Raises:
        ValueError: If num_bytes is negative.
    """
    if num_bytes < 0:
        raise ValueError("num_bytes must be non-negative")

    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"
