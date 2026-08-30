"""Tests for src.file_loader.

All test datasets are built in-memory so the tests do not depend on any
private/local files and can run anywhere (including CI).
"""

from __future__ import annotations

import io

import pandas as pd
import pytest

from src.file_loader import (
    FileLoadError,
    get_excel_sheet_names,
    load_csv,
    load_dataset,
    load_excel,
)


# --- CSV: happy path ----------------------------------------------------------


def test_load_csv_clean_comma_delimited() -> None:
    content = b"id,name,amount\n1,Alice,10.5\n2,Bob,20\n3,Carol,30\n"
    result = load_csv(content, "clean.csv")

    assert result.file_type == "csv"
    assert result.delimiter_used == ","
    assert list(result.dataframe.columns) == ["id", "name", "amount"]
    assert result.dataframe.shape == (3, 3)
    assert result.warnings == []


def test_load_csv_semicolon_delimiter_is_detected() -> None:
    content = "id;name;amount\n1;Alice;10.5\n2;Bob;20\n".encode("utf-8")
    result = load_csv(content, "semicolon.csv")

    assert result.delimiter_used == ";"
    assert result.dataframe.shape == (2, 3)


def test_load_csv_hebrew_content_is_decoded() -> None:
    text = "id,name\n1,שלום\n2,עולם\n"
    content = text.encode("windows-1255")
    result = load_csv(content, "hebrew.csv")

    assert result.dataframe.loc[0, "name"] == "שלום"
    assert result.encoding_used in ("windows-1255", "cp1255")


# --- CSV: edge cases and common problems --------------------------------------


def test_load_csv_empty_file_raises() -> None:
    with pytest.raises(FileLoadError, match="empty"):
        load_csv(b"", "empty.csv")


def test_load_csv_whitespace_only_file_raises() -> None:
    with pytest.raises(FileLoadError):
        load_csv(b"   \n   \n", "blank.csv")


def test_load_csv_duplicate_column_names_are_deduped() -> None:
    content = b"id,amount,amount\n1,10,20\n2,30,40\n"
    result = load_csv(content, "dupes.csv")

    assert list(result.dataframe.columns) == ["id", "amount", "amount_1"]
    assert any("Duplicate column name" in w for w in result.warnings)


def test_load_csv_headerless_file_produces_warning() -> None:
    # First row looks like data (all numeric), not headers.
    content = b"1,2,3\n4,5,6\n7,8,9\n"
    result = load_csv(content, "headerless.csv")

    assert any("does not look like column headers" in w for w in result.warnings)


def test_load_csv_single_column_wrong_delimiter_warns() -> None:
    # Force a delimiter that does not match the actual data.
    content = b"id,name\n1,Alice\n2,Bob\n"
    result = load_csv(content, "forced.csv", delimiter=";")

    assert result.dataframe.shape[1] == 1
    assert any("Only one column was detected" in w for w in result.warnings)


def test_load_csv_file_too_large_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.file_loader.MAX_FILE_SIZE_MB", 0)
    content = b"id,name\n1,Alice\n"

    with pytest.raises(FileLoadError, match="exceeds"):
        load_csv(content, "toolarge.csv")


def test_load_csv_accepts_file_like_object() -> None:
    content = b"id,name\n1,Alice\n2,Bob\n"
    buffer = io.BytesIO(content)
    buffer.name = "buffer.csv"

    result = load_csv(buffer, "buffer.csv")

    assert result.dataframe.shape == (2, 2)
    # The buffer must remain readable afterwards (seek(0) restored).
    assert buffer.read() == content


# --- Excel ---------------------------------------------------------------------


def _make_excel_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    return buffer.getvalue()


def test_load_excel_single_sheet() -> None:
    df = pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]})
    content = _make_excel_bytes({"Sheet1": df})

    result = load_excel(content, "single.xlsx")

    assert result.file_type == "excel"
    assert result.sheet_name == "Sheet1"
    assert result.available_sheets == ["Sheet1"]
    assert result.dataframe.shape == (2, 2)


def test_get_excel_sheet_names_lists_all_sheets() -> None:
    content = _make_excel_bytes(
        {
            "Orders": pd.DataFrame({"id": [1]}),
            "Customers": pd.DataFrame({"id": [1]}),
        }
    )

    sheet_names = get_excel_sheet_names(content, "multi.xlsx")

    assert sheet_names == ["Orders", "Customers"]


def test_load_excel_specific_sheet_is_selected() -> None:
    content = _make_excel_bytes(
        {
            "Orders": pd.DataFrame({"order_id": [1, 2]}),
            "Customers": pd.DataFrame({"customer_id": [10, 20, 30]}),
        }
    )

    result = load_excel(content, "multi.xlsx", sheet_name="Customers")

    assert result.sheet_name == "Customers"
    assert result.dataframe.shape == (3, 1)


def test_load_excel_unknown_sheet_raises() -> None:
    content = _make_excel_bytes({"Sheet1": pd.DataFrame({"id": [1]})})

    with pytest.raises(FileLoadError, match="was not found"):
        load_excel(content, "single.xlsx", sheet_name="DoesNotExist")


def test_load_excel_empty_file_raises() -> None:
    with pytest.raises(FileLoadError, match="empty"):
        load_excel(b"", "empty.xlsx")


def test_load_excel_invalid_content_raises() -> None:
    with pytest.raises(FileLoadError, match="Could not open"):
        load_excel(b"this is not a real excel file", "fake.xlsx")


# --- load_dataset dispatcher ----------------------------------------------------


def test_load_dataset_dispatches_csv_by_extension() -> None:
    content = b"id,name\n1,Alice\n"
    result = load_dataset(content, "data.csv")

    assert result.file_type == "csv"


def test_load_dataset_dispatches_excel_by_extension() -> None:
    content = _make_excel_bytes({"Sheet1": pd.DataFrame({"id": [1]})})
    result = load_dataset(content, "data.xlsx")

    assert result.file_type == "excel"


def test_load_dataset_unsupported_extension_raises() -> None:
    with pytest.raises(FileLoadError, match="Unsupported file type"):
        load_dataset(b"whatever", "data.pdf")


def test_load_dataset_missing_filename_raises() -> None:
    with pytest.raises(ValueError):
        load_dataset(b"id,name\n1,Alice\n")
