"""Robust CSV/Excel file loading.

This module is the single entry point for turning an unfamiliar, uploaded
CSV or Excel file into a pandas DataFrame plus metadata about how it was
parsed (encoding, delimiter, sheet, warnings raised along the way).

Design goals:
    * Never crash on a malformed file -- raise a clear ``FileLoadError`` with
      an actionable message instead.
    * Never silently guess in a way that hides a problem -- anything
      surprising (deduped columns, missing headers, single-column parses)
      is recorded in ``LoadedDataset.warnings`` so the UI/audit trail can
      surface it.
    * Work with plain paths, raw bytes, or file-like objects (e.g. a
      Streamlit ``UploadedFile``), so the same code path is used in tests,
      scripts, and the app.
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Optional, Union

import chardet
import pandas as pd

from src.config import (
    CSV_DELIMITER_CANDIDATES,
    CSV_EXTENSIONS,
    DEFAULT_CSV_DELIMITER,
    ENCODING_CONFIDENCE_THRESHOLD,
    ENCODING_FALLBACKS,
    EXCEL_EXTENSIONS,
    MAX_FILE_SIZE_MB,
    SNIFF_SAMPLE_SIZE,
    SUPPORTED_EXTENSIONS,
)

FileSource = Union[str, Path, bytes, BinaryIO]


class FileLoadError(Exception):
    """Raised when a file cannot be turned into a usable DataFrame.

    The message is written to be shown directly to an end user in the
    Streamlit UI, so it should always explain what went wrong and, where
    possible, how to fix it.
    """


@dataclass
class LoadedDataset:
    """A successfully parsed dataset plus metadata about how it was parsed."""

    dataframe: pd.DataFrame
    file_name: str
    file_type: str  # "csv" or "excel"
    sheet_name: Optional[str] = None
    available_sheets: list[str] = field(default_factory=list)
    encoding_used: Optional[str] = None
    delimiter_used: Optional[str] = None
    header_row: Optional[int] = 0
    warnings: list[str] = field(default_factory=list)


# --- Internal helpers --------------------------------------------------------


def _read_file_bytes(file: FileSource, filename: Optional[str]) -> tuple[bytes, str]:
    """Read raw bytes from a path, bytes object, or file-like object.

    Args:
        file: A path, raw bytes, or a file-like object (must support
            ``.read()``; ``.seek()`` is used to leave it re-readable when
            present, e.g. a Streamlit ``UploadedFile``).
        filename: Explicit filename to use. Required when ``file`` does not
            carry its own name (raw bytes or a generic buffer).

    Returns:
        A tuple of (raw_bytes, resolved_filename).

    Raises:
        ValueError: If no filename can be determined.
    """
    if isinstance(file, (str, Path)):
        path = Path(file)
        return path.read_bytes(), filename or path.name

    if isinstance(file, bytes):
        if not filename:
            raise ValueError("filename is required when passing raw bytes")
        return file, filename

    name = filename or getattr(file, "name", None)
    if not name:
        raise ValueError(
            "filename could not be determined from the file object; pass filename explicitly"
        )

    if hasattr(file, "seek"):
        file.seek(0)
    data = file.read()
    if hasattr(file, "seek"):
        file.seek(0)
    return data, name


def detect_encoding(raw_bytes: bytes) -> str:
    """Guess a text encoding for the given bytes.

    Falls back to "utf-8" when detection confidence is too low to trust,
    since that is the most common encoding and produces a clear decode
    error later if wrong (which is then retried against
    ``ENCODING_FALLBACKS``).

    Args:
        raw_bytes: Raw file content.

    Returns:
        A codec name suitable for ``bytes.decode()``.
    """
    if not raw_bytes:
        return "utf-8"

    sample = raw_bytes[:SNIFF_SAMPLE_SIZE]
    result = chardet.detect(sample)
    encoding = (result.get("encoding") or "utf-8").lower()
    confidence = result.get("confidence") or 0.0

    if confidence < ENCODING_CONFIDENCE_THRESHOLD:
        return "utf-8"
    if encoding == "ascii":
        return "utf-8"
    return encoding


def _decode_with_fallbacks(raw_bytes: bytes, preferred_encoding: str) -> tuple[str, str]:
    """Try to decode bytes, starting with the preferred encoding.

    Args:
        raw_bytes: Raw file content.
        preferred_encoding: Encoding to try first (usually auto-detected or
            user-supplied).

    Returns:
        A tuple of (decoded_text, encoding_used).

    Raises:
        FileLoadError: If no candidate encoding can decode the content.
    """
    candidates = [preferred_encoding, *[e for e in ENCODING_FALLBACKS if e != preferred_encoding]]
    for candidate in candidates:
        try:
            return raw_bytes.decode(candidate), candidate
        except (UnicodeDecodeError, LookupError):
            continue
    raise FileLoadError(
        f"Could not decode the file using any supported encoding: {', '.join(candidates)}."
    )


def detect_delimiter(sample_text: str) -> str:
    """Guess the CSV delimiter from a text sample.

    Uses ``csv.Sniffer`` first, falling back to counting candidate
    delimiter occurrences on the first line when sniffing fails (e.g. on a
    single-column or very short file).

    Args:
        sample_text: A text sample, ideally including several full lines.

    Returns:
        The detected delimiter character.
    """
    try:
        dialect = csv.Sniffer().sniff(sample_text, delimiters="".join(CSV_DELIMITER_CANDIDATES))
        return dialect.delimiter
    except csv.Error:
        pass

    lines = sample_text.splitlines()
    first_line = lines[0] if lines else ""
    counts = {d: first_line.count(d) for d in CSV_DELIMITER_CANDIDATES}
    best_delimiter = max(counts, key=counts.get)
    return best_delimiter if counts[best_delimiter] > 0 else DEFAULT_CSV_DELIMITER


def _looks_headerless(columns: list[Any]) -> bool:
    """Heuristic check for whether parsed "column names" are really data.

    Triggers when most column labels look pandas-generated (``Unnamed: N``),
    blank, or purely numeric -- all signs the first data row was consumed
    as a header when the file actually has none.

    Args:
        columns: Column labels as parsed by pandas.

    Returns:
        True if the columns look like they are not real headers.
    """
    if not columns:
        return False
    generic_count = sum(
        1
        for col in columns
        if str(col).lower().startswith("unnamed:")
        or str(col).strip() == ""
        or str(col).strip().replace(".", "", 1).isdigit()
    )
    return generic_count / len(columns) > 0.5


_PANDAS_MANGLE_RE = re.compile(r"^(.*)\.(\d+)$")


def _unmangle_pandas_duplicates(columns: list[str]) -> list[str]:
    """Reverse pandas' automatic ".1", ".2" duplicate-column suffixing.

    Pandas renames duplicate header names itself while parsing (e.g. two
    "amount" columns become "amount" and "amount.1"), which would otherwise
    hide the duplication from ``_dedupe_columns`` and skip the warning.
    This restores the shared base name so duplicates are re-detected and
    renamed consistently (with an "_N" suffix) by ``_dedupe_columns``.

    Args:
        columns: Column labels as parsed by pandas.

    Returns:
        Column labels with pandas' ".N" mangling reversed where it appears
        to correspond to a real duplicate of an existing base name.
    """
    base_names = {col for col in columns if not _PANDAS_MANGLE_RE.match(col)}
    result = []
    for col in columns:
        match = _PANDAS_MANGLE_RE.match(col)
        if match and match.group(1) in base_names:
            result.append(match.group(1))
        else:
            result.append(col)
    return result


def _dedupe_columns(columns: list[Any]) -> tuple[list[str], list[str]]:
    """Rename duplicate/blank column labels so every column name is unique.

    Args:
        columns: Column labels as parsed by pandas.

    Returns:
        A tuple of (new_column_names, warning_messages).
    """
    warnings: list[str] = []
    seen: dict[str, int] = {}
    new_columns: list[str] = []

    for raw in columns:
        col = str(raw).strip()
        if col == "" or col.lower().startswith("unnamed:"):
            col = "column"

        if col in seen:
            seen[col] += 1
            new_name = f"{col}_{seen[col]}"
            warnings.append(f"Duplicate column name '{col}' was renamed to '{new_name}'.")
            new_columns.append(new_name)
        else:
            seen[col] = 0
            new_columns.append(col)

    return new_columns, warnings


def _check_size_limit(raw_bytes: bytes, name: str) -> None:
    """Raise FileLoadError if the file exceeds MAX_FILE_SIZE_MB."""
    size_mb = len(raw_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise FileLoadError(
            f"'{name}' is {size_mb:.1f} MB, which exceeds the {MAX_FILE_SIZE_MB} MB limit for this app."
        )


# --- Public API ---------------------------------------------------------------


def load_csv(
    file: FileSource,
    filename: Optional[str] = None,
    *,
    encoding: Optional[str] = None,
    delimiter: Optional[str] = None,
    header: Optional[int] = 0,
) -> LoadedDataset:
    """Load a CSV (or delimited text) file into a LoadedDataset.

    Args:
        file: A path, raw bytes, or file-like object.
        filename: Explicit filename, required if it cannot be inferred from
            ``file``.
        encoding: Force a specific encoding instead of auto-detecting.
        delimiter: Force a specific delimiter instead of auto-detecting.
        header: Row index to use as column names, or None for headerless
            files (pandas assigns integer column names in that case).

    Returns:
        A LoadedDataset with the parsed DataFrame and parsing metadata.

    Raises:
        FileLoadError: If the file is empty, too large, undecodable, or
            cannot be parsed as tabular data.
    """
    raw_bytes, name = _read_file_bytes(file, filename)

    if len(raw_bytes) == 0:
        raise FileLoadError(f"'{name}' is empty (0 bytes). Please upload a non-empty file.")

    _check_size_limit(raw_bytes, name)

    used_encoding = encoding or detect_encoding(raw_bytes)
    text, used_encoding = _decode_with_fallbacks(raw_bytes, used_encoding)

    if not text.strip():
        raise FileLoadError(f"'{name}' contains no readable data (file is blank after decoding).")

    used_delimiter = delimiter or detect_delimiter(text[:SNIFF_SAMPLE_SIZE])

    try:
        df = pd.read_csv(io.StringIO(text), sep=used_delimiter, header=header, engine="python")
    except pd.errors.EmptyDataError as exc:
        raise FileLoadError(f"'{name}' has no columns to parse (empty or header-only file).") from exc
    except pd.errors.ParserError as exc:
        raise FileLoadError(
            f"Could not parse '{name}' as CSV using delimiter '{used_delimiter}': {exc}"
        ) from exc

    warnings: list[str] = []

    if df.shape[1] == 1 and used_delimiter != DEFAULT_CSV_DELIMITER:
        warnings.append(
            f"Only one column was detected using delimiter '{used_delimiter}'. "
            "The file may use a different delimiter than expected."
        )

    if header is not None and _looks_headerless(df.columns.tolist()):
        warnings.append(
            "The first row does not look like column headers (it looks like data). "
            "Reload with header=None if this file has no header row."
        )

    raw_columns = [str(c) for c in df.columns]
    if header is not None:
        raw_columns = _unmangle_pandas_duplicates(raw_columns)
    new_columns, dedupe_warnings = _dedupe_columns(raw_columns)
    df.columns = new_columns
    warnings.extend(dedupe_warnings)

    if df.shape[0] == 0:
        warnings.append("The file was parsed successfully but contains 0 data rows.")

    return LoadedDataset(
        dataframe=df,
        file_name=name,
        file_type="csv",
        encoding_used=used_encoding,
        delimiter_used=used_delimiter,
        header_row=header,
        warnings=warnings,
    )


def get_excel_sheet_names(file: FileSource, filename: Optional[str] = None) -> list[str]:
    """List the sheet names available in an Excel file.

    Intended to be called before ``load_excel`` so the caller (e.g. the
    Streamlit UI) can let the user pick a sheet when there is more than one.

    Args:
        file: A path, raw bytes, or file-like object.
        filename: Explicit filename, required if it cannot be inferred.

    Returns:
        List of sheet names in workbook order.

    Raises:
        FileLoadError: If the file is empty or not a valid Excel workbook.
    """
    raw_bytes, name = _read_file_bytes(file, filename)

    if len(raw_bytes) == 0:
        raise FileLoadError(f"'{name}' is empty (0 bytes). Please upload a non-empty file.")

    try:
        with pd.ExcelFile(io.BytesIO(raw_bytes)) as xls:
            return list(xls.sheet_names)
    except Exception as exc:
        raise FileLoadError(f"Could not open '{name}' as an Excel file: {exc}") from exc


def load_excel(
    file: FileSource,
    filename: Optional[str] = None,
    *,
    sheet_name: Optional[Union[str, int]] = None,
    header: Optional[int] = 0,
) -> LoadedDataset:
    """Load one sheet of an Excel workbook into a LoadedDataset.

    Args:
        file: A path, raw bytes, or file-like object.
        filename: Explicit filename, required if it cannot be inferred.
        sheet_name: Sheet to load. Defaults to the first sheet when omitted.
        header: Row index to use as column names, or None for headerless
            sheets.

    Returns:
        A LoadedDataset with the parsed DataFrame and parsing metadata.

    Raises:
        FileLoadError: If the file is empty, too large, not a valid
            workbook, has no sheets, or the requested sheet does not exist.
    """
    raw_bytes, name = _read_file_bytes(file, filename)

    if len(raw_bytes) == 0:
        raise FileLoadError(f"'{name}' is empty (0 bytes). Please upload a non-empty file.")

    _check_size_limit(raw_bytes, name)

    try:
        xls = pd.ExcelFile(io.BytesIO(raw_bytes))
    except Exception as exc:
        raise FileLoadError(f"Could not open '{name}' as an Excel file: {exc}") from exc

    available_sheets = list(xls.sheet_names)
    if not available_sheets:
        raise FileLoadError(f"'{name}' does not contain any sheets.")

    chosen_sheet = sheet_name if sheet_name is not None else available_sheets[0]
    if chosen_sheet not in available_sheets:
        raise FileLoadError(
            f"Sheet '{chosen_sheet}' was not found in '{name}'. Available sheets: {available_sheets}."
        )

    try:
        df = pd.read_excel(xls, sheet_name=chosen_sheet, header=header)
    except Exception as exc:
        raise FileLoadError(f"Could not parse sheet '{chosen_sheet}' in '{name}': {exc}") from exc

    warnings: list[str] = []

    if header is not None and _looks_headerless(df.columns.tolist()):
        warnings.append(
            "The first row does not look like column headers (it looks like data). "
            "Reload with header=None if this sheet has no header row."
        )

    raw_columns = [str(c) for c in df.columns]
    if header is not None:
        raw_columns = _unmangle_pandas_duplicates(raw_columns)
    new_columns, dedupe_warnings = _dedupe_columns(raw_columns)
    df.columns = new_columns
    warnings.extend(dedupe_warnings)

    if df.shape[0] == 0:
        warnings.append("The selected sheet was parsed successfully but contains 0 data rows.")

    return LoadedDataset(
        dataframe=df,
        file_name=name,
        file_type="excel",
        sheet_name=str(chosen_sheet),
        available_sheets=[str(s) for s in available_sheets],
        header_row=header,
        warnings=warnings,
    )


def load_dataset(
    file: FileSource,
    filename: Optional[str] = None,
    **kwargs: Any,
) -> LoadedDataset:
    """Dispatch to load_csv or load_excel based on the file extension.

    Args:
        file: A path, raw bytes, or file-like object.
        filename: Explicit filename, required if it cannot be inferred.
        **kwargs: Forwarded to load_csv (encoding, delimiter, header) or
            load_excel (sheet_name, header) depending on the detected type.

    Returns:
        A LoadedDataset with the parsed DataFrame and parsing metadata.

    Raises:
        FileLoadError: If the extension is unsupported or parsing fails.
        ValueError: If no filename can be determined.
    """
    name = filename or getattr(file, "name", None)
    if not name:
        raise ValueError("filename could not be determined; pass filename explicitly.")

    ext = Path(name).suffix.lower()

    if ext in CSV_EXTENSIONS:
        return load_csv(
            file,
            name,
            encoding=kwargs.get("encoding"),
            delimiter=kwargs.get("delimiter"),
            header=kwargs.get("header", 0),
        )

    if ext in EXCEL_EXTENSIONS:
        return load_excel(
            file,
            name,
            sheet_name=kwargs.get("sheet_name"),
            header=kwargs.get("header", 0),
        )

    raise FileLoadError(
        f"Unsupported file type '{ext}' for '{name}'. Supported types: {', '.join(SUPPORTED_EXTENSIONS)}."
    )
