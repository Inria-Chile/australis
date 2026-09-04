#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow.dataset as ds
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from polarfunc.annotation import build_unigene_master
from polarfunc.provenance import sha256_file, utc_now, write_json, write_task_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the one-row-per-CDHit ACE unigene master.")
    project = ROOT.parent
    parser.add_argument("--input", type=Path, default=project / "data" / "Annotation_Table_AGN_CDH_Tax_KEGG_EGG.tsv.gz")
    parser.add_argument("--source-input", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "master" / "unigene_master")
    parser.add_argument("--chunksize", type=int, default=5_000_000)
    parser.add_argument("--max-chunks", type=int)
    parser.add_argument("--task-id", default="P05")
    args = parser.parse_args()
    started = utc_now()
    registry = yaml.safe_load((ROOT / "config" / "schema_registry.yaml").read_text(encoding="utf-8"))
    functional_columns = registry["annotation_functional_fields"]
    summary = build_unigene_master(
        args.input,
        args.output_dir,
        functional_columns=functional_columns,
        chunksize=args.chunksize,
        max_chunks=args.max_chunks,
    )
    dataset = ds.dataset(args.output_dir, format="parquet")
    output_rows = dataset.count_rows()
    summary["dataset_rows"] = output_rows
    summary["functional_columns"] = functional_columns
    expected = 89_739_060
    full_run = args.max_chunks is None
    validation = {
        "passed": output_rows == summary["representative_rows"] and (not full_run or output_rows == expected),
        "checks": {
            "parts_match_summary": output_rows == summary["representative_rows"],
            "expected_unigenes_exact": output_rows == expected if full_run else None,
            "probe_only": not full_run,
        },
    }
    summary["validation"] = validation
    report_path = ROOT / "reports" / ("unigene_master_validation.json" if full_run else "unigene_master_probe.json")
    write_json(report_path, summary)
    source = args.source_input or args.input
    inputs = [{"path": str(source), "size_bytes": source.stat().st_size, "sha256": sha256_file(source) if full_run else None}]
    outputs = [
        {"path": part["path"], "size_bytes": part["size_bytes"], "rows": part["rows"], "sha256": sha256_file(Path(part["path"]))}
        for part in summary["parts"]
    ]
    outputs.append({"path": str(report_path), "size_bytes": report_path.stat().st_size, "sha256": sha256_file(report_path)})
    write_task_manifest(
        ROOT,
        args.task_id,
        started_at=started,
        command=" ".join(sys.argv),
        inputs=inputs,
        outputs=outputs,
        parameters={"chunksize": args.chunksize, "max_chunks": args.max_chunks, "functional_columns": functional_columns},
        validation=validation,
    )
    print(json.dumps({"input_rows": summary["input_rows"], "representative_rows": output_rows, "elapsed_seconds": summary["elapsed_seconds"], "validation": validation}, indent=2))
    return 0 if validation["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

