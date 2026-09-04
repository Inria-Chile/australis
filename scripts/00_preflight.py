#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from polarfunc.provenance import collect_preflight, sha256_file, utc_now, write_json, write_task_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the Grid5000 CPU execution context.")
    parser.add_argument("--project-root", type=Path, default=ROOT.parent)
    args = parser.parse_args()
    started = utc_now()
    report = collect_preflight(args.project_root, ROOT)
    json_path = ROOT / "reports" / "preflight.json"
    md_path = ROOT / "reports" / "preflight.md"
    write_json(json_path, report)
    checks = report["validation"]["checks"]
    markdown = [
        "# Grid'5000 preflight",
        "",
        f"- Host: `{report['hostname']}`",
        f"- OAR job: `{report['oar']['job_id']}`",
        f"- Allocated nodes: `{', '.join(report['oar']['allocated_nodes'])}`",
        f"- Logical CPUs: `{report['resources']['logical_cpus']}`",
        f"- RAM: `{report['resources']['memory_total_bytes'] / 1024**4:.2f} TiB`",
        f"- Project free space: `{report['resources']['project_disk_free_bytes'] / 1024**3:.1f} GiB`",
        f"- Python: `{report['python']['executable']}`",
        f"- Environment: `{report['environment_prefix']}`",
        "",
        "## Validation",
        "",
        *[f"- {name}: `{'PASS' if passed else 'FAIL'}`" for name, passed in checks.items()],
        "",
        "## Warnings",
        "",
        *([f"- {warning}" for warning in report["warnings"]] or ["- None"]),
    ]
    md_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    outputs = [
        {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        for path in [json_path, md_path]
    ]
    write_task_manifest(
        ROOT,
        "P00",
        started_at=started,
        command=" ".join(sys.argv),
        inputs=[],
        outputs=outputs,
        parameters={"project_root": str(args.project_root)},
        validation=report["validation"],
    )
    print(json.dumps(report["validation"], indent=2))
    return 0 if report["validation"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

