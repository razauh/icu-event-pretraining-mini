"""eICU demo table loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Iterator

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

TABLE_FILENAMES = {
    "patient": "patient",
    "apachePatientResult": "apachePatientResult",
    "diagnosis": "diagnosis",
    "lab": "lab",
    "medication": "medication",
    "infusionDrug": "infusiondrug",
    "treatment": "treatment",
    "vitalPeriodic": "vitalPeriodic",
    "vitalAperiodic": "vitalAperiodic",
}

REQUIRED_COLUMNS = {
    "patient": frozenset({
        "patientunitstayid",
        "uniquepid",
        "hospitalid",
        "hospitaldischargestatus",
        "unitdischargeoffset",
    }),
    "apachePatientResult": frozenset({"patientunitstayid"}),
    "diagnosis": frozenset({"patientunitstayid", "diagnosisstring", "diagnosisoffset"}),
    "lab": frozenset({"patientunitstayid", "labname", "labresult", "labresultoffset"}),
    "medication": frozenset({"patientunitstayid", "drugname", "drughiclseqno", "drugstartoffset"}),
    "infusionDrug": frozenset({"patientunitstayid", "drugname", "infusionoffset"}),
    "treatment": frozenset({"patientunitstayid", "treatmentstring", "treatmentoffset"}),
    "vitalPeriodic": frozenset({"patientunitstayid", "observationoffset"}),
    "vitalAperiodic": frozenset({"patientunitstayid", "observationoffset"}),
}


def discover_raw_dir(raw_dir: str | Path) -> Path:
    root = Path(raw_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"raw directory does not exist: {root}")
    for name in TABLE_FILENAMES.values():
        if (root / f"{name}.csv.gz").is_file() or (root / f"{name}.csv").is_file():
            return root
    if root.name == "2.0.1" and root.parent.name == "eicu-crd-demo":
        return root
    matches = []
    try:
        for p in root.rglob("2.0.1"):
            if p.is_dir() and p.parent.name == "eicu-crd-demo":
                matches.append(p)
    except Exception:
        pass
    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        paths = ", ".join(str(m) for m in sorted(matches))
        raise ValueError(f"ambiguous raw directory: found multiple eicu-crd-demo/2.0.1 subdirectories: {paths}")
    else:
        raise FileNotFoundError(f"could not discover eicu-crd-demo/2.0.1 directory in {root}")


def table_path(raw_dir: str | Path, table_name: str) -> Path:
    """Return the expected CSV path for an eICU demo table."""
    try:
        resolved_dir = discover_raw_dir(raw_dir)
    except Exception:
        resolved_dir = Path(raw_dir)
    filename = TABLE_FILENAMES.get(table_name, table_name)
    gz_path = resolved_dir / f"{filename}.csv.gz"
    csv_path = resolved_dir / f"{filename}.csv"
    if gz_path.is_file():
        return gz_path
    if csv_path.is_file():
        return csv_path
    return gz_path


def load_table_chunks(
    raw_dir: str | Path,
    table_name: str,
    chunksize: int = 50000,
    columns: Sequence[str] | None = None,
) -> Iterator[pd.DataFrame]:
    resolved_dir = discover_raw_dir(raw_dir)
    path = table_path(resolved_dir, table_name)
    if not path.is_file():
        raise FileNotFoundError(f"required eICU demo table file is missing: {path}")
    try:
        header = pd.read_csv(path, nrows=0)
    except pd.errors.EmptyDataError as error:
        raise ValueError(f"eICU demo table '{table_name}' is empty and has no columns: {path}") from error
    except pd.errors.ParserError as error:
        raise ValueError(f"eICU demo table '{table_name}' could not be parsed: {path}") from error
    if table_name in REQUIRED_COLUMNS:
        missing = REQUIRED_COLUMNS[table_name].difference(header.columns)
        if missing:
            raise ValueError(
                f"eICU demo table '{table_name}' is missing required columns: {', '.join(sorted(missing))}"
            )
    usecols = list(columns) if columns is not None else None
    def chunk_generator():
        try:
            reader = pd.read_csv(path, chunksize=chunksize, usecols=usecols)
            for chunk in reader:
                yield chunk
        except pd.errors.ParserError as error:
            raise ValueError(f"eICU demo table '{table_name}' could not be parsed: {path}") from error
    return chunk_generator()


def load_table_full(
    raw_dir: str | Path,
    table_name: str,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    resolved_dir = discover_raw_dir(raw_dir)
    path = table_path(resolved_dir, table_name)
    if not path.is_file():
        raise FileNotFoundError(f"required eICU demo table file is missing: {path}")
    try:
        header = pd.read_csv(path, nrows=0)
    except pd.errors.EmptyDataError as error:
        raise ValueError(f"eICU demo table '{table_name}' is empty and has no columns: {path}") from error
    except pd.errors.ParserError as error:
        raise ValueError(f"eICU demo table '{table_name}' could not be parsed: {path}") from error
    if table_name in REQUIRED_COLUMNS:
        missing = REQUIRED_COLUMNS[table_name].difference(header.columns)
        if missing:
            raise ValueError(
                f"eICU demo table '{table_name}' is missing required columns: {', '.join(sorted(missing))}"
            )
    usecols = list(columns) if columns is not None else None
    try:
        return pd.read_csv(path, usecols=usecols)
    except pd.errors.ParserError as error:
        raise ValueError(f"eICU demo table '{table_name}' could not be parsed: {path}") from error


def load_tables(
    raw_dir: str | Path, tables: Sequence[str] = MVP_TABLES
) -> dict[str, pd.DataFrame]:
    """Load and validate selected eICU demo CSV tables without writing data."""
    selected_tables = list(tables)
    unknown_tables = [name for name in selected_tables if name not in MVP_TABLES]
    if unknown_tables:
        names = ", ".join(str(name) for name in unknown_tables)
        raise ValueError(f"unknown eICU demo table name(s): {names}")
    resolved_dir = discover_raw_dir(raw_dir)
    loaded: dict[str, pd.DataFrame] = {}
    for table_name in selected_tables:
        is_optional = (table_name == "apachePatientResult")
        try:
            path = table_path(resolved_dir, table_name)
            if not path.is_file():
                if is_optional:
                    continue
                raise FileNotFoundError(
                    f"required eICU demo table file is missing: {path}"
                )
            loaded[table_name] = load_table_full(resolved_dir, table_name)
        except FileNotFoundError as error:
            if is_optional:
                continue
            raise error
    return loaded
