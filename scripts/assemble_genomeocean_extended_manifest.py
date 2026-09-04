#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extended-manifest", type=Path, required=True)
    parser.add_argument("--core-dir", type=Path, required=True)
    parser.add_argument("--remainder-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--report-md", type=Path, required=True)
    args = parser.parse_args()
    output_manifest = args.output_dir / "embedding_manifest.parquet"
    if any(path.exists() for path in (output_manifest, args.report_json, args.report_md)):
        raise SystemExit("Refusing to overwrite GenomeOcean extended assembly")

    expected = set(pd.read_parquet(args.extended_manifest, columns=["CDHit_ID"]).CDHit_ID.astype(str))
    parts = []
    for source, directory in (("core", args.core_dir), ("remainder", args.remainder_dir)):
        frame = pd.read_parquet(directory / "embedding_manifest.parquet")
        frame["source"] = source
        frame["source_dir"] = str(directory.resolve())
        parts.append(frame)
    combined = pd.concat(parts, ignore_index=True).sort_values("CDHit_ID").reset_index(drop=True)
    source_rows = combined.groupby("source").size().to_dict()
    checks = {
        "expected_total_92272": len(expected) == 92_272,
        "embedded_ids_unique": combined.CDHit_ID.is_unique,
        "embedded_ids_exact": set(combined.CDHit_ID.astype(str)) == expected,
        "source_rows_exact": source_rows == {"core": 50_000, "remainder": 42_272},
    }
    if not all(checks.values()):
        raise SystemExit(f"GenomeOcean assembly failed: {checks}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_manifest, index=False, compression="zstd")
    report = {
        "status": "PASS", "expected_total": len(expected), "embedded": len(combined),
        "excluded_context": 0, "source_rows": source_rows, "checks": checks,
        "embedding_manifest": str(output_manifest.resolve()),
        "embedding_manifest_sha256": sha256(output_manifest),
        "extended_manifest_sha256": sha256(args.extended_manifest),
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2) + "\n")
    args.report_md.write_text(
        "# GenomeOcean extended validation\n\n"
        f"- Status: **PASS**\n- Embedded: {len(combined):,}/{len(expected):,}\n"
        f"- Core rows referenced: {source_rows['core']:,}\n"
        f"- Remainder rows referenced: {source_rows['remainder']:,}\n"
        "- Context exclusions: 0\n- Core embeddings were not recomputed.\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
