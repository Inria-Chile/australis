#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from polarfunc.provenance import sha256_file, utc_now, write_json, write_task_manifest

URL = "https://github.com/EmileFaure/ACE_gene_centric_scripts.git"
COMMIT = "8dd822eecbe6fb582cca9375c3c343577a4b736a"


def run(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(args, cwd=cwd, text=True, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(f"Command failed: {' '.join(args)}\n{result.stderr}")
    return result.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture the immutable Faure reference implementation.")
    parser.add_argument("--destination", type=Path, default=ROOT / "references" / "faure" / "ACE_gene_centric_scripts")
    args = parser.parse_args()
    started = utc_now()
    destination = args.destination
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        run("git", "clone", "--filter=blob:none", URL, str(destination))
    run("git", "fetch", "origin", COMMIT, cwd=destination)
    run("git", "checkout", "--detach", COMMIT, cwd=destination)
    head = run("git", "rev-parse", "HEAD", cwd=destination)
    remotes = run("git", "remote", "-v", cwd=destination)
    scripts = []
    for path in sorted(destination.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".r", ".py", ".sh", ".pbs"}:
            scripts.append({"path": str(path.relative_to(destination)), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    report = {
        "url": URL,
        "requested_commit": COMMIT,
        "head": head,
        "remotes": remotes.splitlines(),
        "scripts": scripts,
        "reference_is_read_only_policy": True,
        "datarmor_scripts_executed": False,
    }
    report_path = ROOT / "reports" / "faure_reference.json"
    write_json(report_path, report)
    notes_path = ROOT / "docs" / "reproduction_notes.md"
    notes_path.write_text(
        "# Fauré reference implementation\n\n"
        f"Captured `{URL}` at immutable commit `{head}`. The original PBS and analysis scripts "
        "are reference material only and were not executed directly. Their physical schemas, "
        "thresholds, and split logic must be transcribed into tested POLAR-FUNC modules.\n",
        encoding="utf-8",
    )
    validation = {"passed": head == COMMIT and bool(scripts), "checks": {"commit_exact": head == COMMIT, "scripts_hashed": bool(scripts)}}
    outputs = [
        {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        for path in [report_path, notes_path]
    ]
    write_task_manifest(
        ROOT,
        "P02",
        started_at=started,
        command=" ".join(sys.argv),
        inputs=[{"url": URL, "commit": COMMIT}],
        outputs=outputs,
        parameters={"destination": str(destination)},
        validation=validation,
    )
    print(json.dumps(validation, indent=2))
    return 0 if validation["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

