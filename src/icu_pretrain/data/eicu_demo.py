"""eICU demo table loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd


MVP_TABLES = [
    "patient",
    "apachePatientResult",
    "diagnosis",
    "lab",
    "medication",
    "infusionDrug",
    "treatment",
    "vitalPeriodic",
    "vitalAperiodic",
]

REQUIRED_COLUMNS = {
    table_name: frozenset({"patientunitstayid"}) for table_name in MVP_TABLES
}


def table_path(raw_dir: str | Path, table_name: str) -> Path:
    """Return the expected CSV path for an eICU demo table."""
    return Path(raw_dir) / f"{table_name}.csv"


def load_tables(
    raw_dir: str | Path, tables: Sequence[str] = MVP_TABLES
) -> dict[str, pd.DataFrame]:
    """Load and validate selected eICU demo CSV tables without writing data."""
    selected_tables = list(tables)
    unknown_tables = [name for name in selected_tables if name not in MVP_TABLES]
    if unknown_tables:
        names = ", ".join(str(name) for name in unknown_tables)
        raise ValueError(f"unknown eICU demo table name(s): {names}")

    loaded: dict[str, pd.DataFrame] = {}
    for table_name in selected_tables:
        path = table_path(raw_dir, table_name)
        if not path.is_file():
            raise FileNotFoundError(
                f"required eICU demo table file is missing: {path}"
            )

        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError as error:
            raise ValueError(
                f"eICU demo table '{table_name}' is empty and has no columns: {path}"
            ) from error
        except pd.errors.ParserError as error:
            raise ValueError(
                f"eICU demo table '{table_name}' could not be parsed: {path}"
            ) from error

        missing_columns = REQUIRED_COLUMNS[table_name].difference(frame.columns)
        if missing_columns:
            columns = ", ".join(sorted(missing_columns))
            raise ValueError(
                f"eICU demo table '{table_name}' is missing required columns: {columns}"
            )
        loaded[table_name] = frame

    return loaded
