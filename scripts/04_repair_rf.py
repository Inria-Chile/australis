#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from polarfunc.provenance import sha256_file, utc_now, write_json, write_task_manifest
from polarfunc.rf import inspect_rf_schema, repair_rf_table, validate_rf_table
from polarfunc.schema import open_text


def raw_examples(path: Path, count: int = 5) -> list[str]:
    with open_text(path) as handle:
        return [handle.readline().rstrip("\r\n") for _ in range(count)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair malformed ACE random-forest tables.")
    project = ROOT.parent
    parser.add_argument("--rf-fl", type=Path, default=project / "data" / "RF_AGC_NZVuniquecut20_T60MAX_FL_All.txt.gz")
    parser.add_argument("--rf-att", type=Path, default=project / "data" / "RF_AGC_NZVuniquecut20_T60MAX_ATT_All.txt.gz")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "rf")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--task-id", default="P04")
    args = parser.parse_args()
    started = utc_now()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {"repair_rationale": "The physical data contain AGC_ID as an initial row-name field omitted from the header.", "fractions": {}}
    outputs: list[dict[str, object]] = []
    inputs: list[dict[str, object]] = []
    all_passed = True
    for fraction, path in [("FL", args.rf_fl), ("ATT", args.rf_att)]:
        inspection = inspect_rf_schema(path)
        frame = repair_rf_table(path, nrows=args.max_rows)
        validation = validate_rf_table(frame)
        output = args.output_dir / f"rf_{fraction}.parquet"
        frame.to_parquet(output, index=False, compression="zstd")
        examples_path = args.output_dir / f"rf_{fraction}_raw_examples.json"
        write_json(examples_path, {"path": str(path), "lines": raw_examples(path)})
        report["fractions"][fraction] = {
            "inspection": inspection.to_dict(),
            "validation": validation,
            "output": str(output),
            "nulls": {column: int(frame[column].isna().sum()) for column in ["Rsquared_rf", "R2_caret", "MSE"]},
        }
        inputs.append({"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path) if args.max_rows is None else None})
        for artifact in [output, examples_path]:
            outputs.append({"path": str(artifact), "size_bytes": artifact.stat().st_size, "sha256": sha256_file(artifact)})
        all_passed &= bool(validation["passed"])
    report_path = ROOT / "reports" / ("rf_repair.json" if args.max_rows is None else "rf_repair_probe.json")
    write_json(report_path, report)
    outputs.append({"path": str(report_path), "size_bytes": report_path.stat().st_size, "sha256": sha256_file(report_path)})
    validation = {"passed": all_passed, "checks": {"both_fractions_valid": all_passed, "probe_only": args.max_rows is not None}}
    write_task_manifest(
        ROOT,
        args.task_id,
        started_at=started,
        command=" ".join(sys.argv),
        inputs=inputs,
        outputs=outputs,
        parameters={"max_rows": args.max_rows, "output_dir": str(args.output_dir)},
        validation=validation,
    )
    print(json.dumps({"validation": validation, "fractions": report["fractions"]}, indent=2))
    return 0 if validation["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

