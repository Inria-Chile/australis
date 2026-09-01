#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from polarfunc_repro.ood_reporting import summarize_ood_runs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate repeated functional OOD runs")
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-repeats", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = summarize_ood_runs(
        args.summaries, bootstrap_repeats=args.bootstrap_repeats, seed=args.seed
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
