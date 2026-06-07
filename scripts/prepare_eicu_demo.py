"""Prepare eICU demo tables into patient-level event streams."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Sequence

from icu_pretrain.constants import EVENT_REPRESENTATIONS
from icu_pretrain.data.eicu_demo import load_tables
from icu_pretrain.data.eicu_event_builder import (
    build_event_streams,
    write_event_streams_jsonl,
)


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for local eICU demo preparation."""
    parser = argparse.ArgumentParser(
        description="Prepare local eICU demo CSVs into patient-level artifacts."
    )
    parser.add_argument(
        "--raw_dir",
        required=True,
        type=Path,
        help="Directory containing the eICU demo CSV tables.",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        type=Path,
        help="Directory for generated processed artifacts.",
    )
    parser.add_argument(
        "--representation",
        choices=EVENT_REPRESENTATIONS,
        default="timegap_static",
        help="Event-stream representation to build (default: timegap_static).",
    )
    parser.add_argument(
        "--min_events_per_stay",
        type=_positive_integer,
        default=5,
        help="Minimum events required to retain a stay (default: 5).",
    )
    return parser


def prepare_eicu_demo(
    raw_dir: Path,
    out_dir: Path,
    representation: str,
    min_events_per_stay: int,
) -> tuple[int, int]:
    """Build and write aggregate processed artifacts under ``out_dir``."""
    tables = load_tables(raw_dir)
    streams, outcomes, stats = build_event_streams(
        tables,
        representation=representation,
        min_events_per_stay=min_events_per_stay,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    write_event_streams_jsonl(out_dir / "event_streams.jsonl", streams)
    outcomes.to_csv(out_dir / "outcomes.csv", index=False)
    with (out_dir / "event_stats.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(stats), handle, indent=2, sort_keys=True)
        handle.write("\n")

    return len(streams), stats.skipped_stays


def main(argv: Sequence[str] | None = None) -> int:
    """Run the preparation command and return its process status."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        kept_stays, skipped_stays = prepare_eicu_demo(
            raw_dir=args.raw_dir,
            out_dir=args.out_dir,
            representation=args.representation,
            min_events_per_stay=args.min_events_per_stay,
        )
    except (FileNotFoundError, OSError, ValueError) as error:
        parser.error(str(error))

    print(
        f"Prepared {kept_stays} event stream(s); skipped {skipped_stays} stay(s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
