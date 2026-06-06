"""eICU demo table loading utilities."""

from pathlib import Path


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


def table_path(raw_dir: str | Path, table_name: str) -> Path:
    """Return the expected CSV path for an eICU demo table."""
    return Path(raw_dir) / f"{table_name}.csv"

