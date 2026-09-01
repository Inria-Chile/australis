#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from polarfunc_repro.release import release_readiness, write_release_readiness


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit the GitHub publication tree")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--archive", action="append", default=[], type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = release_readiness(args.root, tuple(args.archive))
    write_release_readiness(result, args.output)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
