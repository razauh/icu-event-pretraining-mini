from __future__ import annotations

import argparse
from pathlib import Path

from icu_pretrain.analysis.plots import make_plots
from icu_pretrain.analysis.tables import make_tables


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-dir", type=Path, default=None)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--figures-dir", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    make_tables(summary_dir=args.summary_dir, run_root=args.run_root)
    make_plots(summary_dir=args.summary_dir, run_root=args.run_root, figures_dir=args.figures_dir)


if __name__ == "__main__":
    main()
