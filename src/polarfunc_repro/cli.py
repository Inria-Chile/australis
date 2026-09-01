from __future__ import annotations

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

from .config import absolute_paths, compose_config, resolved_yaml
from .download import download_manifest, fetch_zenodo_manifest
from .inventory import write_inventory
from .manifest import load_manifest, verify_manifest, write_verification
from .provenance import capture_provenance, write_provenance
from .reports import summarize_json_reports
from .sharding import build_file_index, write_jsonl_index


def _config(args: argparse.Namespace):
    return compose_config(Path(args.config_dir), args.config_name, args.overrides)


def cmd_compose(args: argparse.Namespace) -> int:
    cfg = _config(args)
    print(resolved_yaml(cfg), end="")
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    cfg = _config(args)
    stages = OmegaConf.to_container(cfg.pipeline.stages, resolve=True)
    print(json.dumps({"project": cfg.project.name, "stages": stages}, indent=2))
    return 0


def cmd_preflight(args: argparse.Namespace) -> int:
    cfg = _config(args)
    forbidden = absolute_paths(cfg)
    output = Path(str(cfg.paths.output_root)) / str(cfg.run.name)
    output.mkdir(parents=True, exist_ok=False)
    config_text = resolved_yaml(cfg)
    (output / "resolved_config.yaml").write_text(config_text, encoding="utf-8")
    write_provenance(capture_provenance(Path.cwd(), config_text), output / "provenance.json")
    result = {"status": "PASS" if not forbidden else "FAIL", "absolute_paths": forbidden}
    (output / "preflight.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 2


def cmd_verify_manifest(args: argparse.Namespace) -> int:
    result = verify_manifest(
        Path(args.manifest), Path(args.root) if args.root else None, workers=args.workers
    )
    if args.output:
        write_verification(result, Path(args.output))
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 2


def cmd_summarize_reports(args: argparse.Namespace) -> int:
    result = summarize_json_reports(Path(args.report_root))
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 2


def cmd_zenodo_manifest(args: argparse.Namespace) -> int:
    manifest = fetch_zenodo_manifest(args.record_id)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "files": len(manifest.artifacts)}, indent=2))
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    paths = download_manifest(load_manifest(Path(args.manifest)), Path(args.destination))
    print(json.dumps({"status": "PASS", "files": [str(path) for path in paths]}, indent=2))
    return 0


def cmd_inventory(args: argparse.Namespace) -> int:
    result = write_inventory([Path(path) for path in args.paths], Path(args.output))
    print(json.dumps({"status": result["status"], "file_count": result["file_count"]}, indent=2))
    return 0


def cmd_bucket_index(args: argparse.Namespace) -> int:
    paths = [Path(path) for path in args.paths]
    for pattern in args.pattern:
        paths.extend(Path(args.root).rglob(pattern))
    records = build_file_index(
        Path(args.root),
        paths,
        bucket_count=args.buckets,
        include_sha256=args.sha256,
    )
    write_jsonl_index(records, Path(args.output))
    print(json.dumps({"status": "PASS", "files": len(records), "buckets": args.buckets}, indent=2))
    return 0


def _add_hydra_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--config-name", default="config")
    parser.add_argument("overrides", nargs="*")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="polarfunc")
    sub = parser.add_subparsers(dest="command", required=True)
    compose_parser = sub.add_parser("compose")
    _add_hydra_arguments(compose_parser)
    compose_parser.set_defaults(func=cmd_compose)
    plan_parser = sub.add_parser("plan")
    _add_hydra_arguments(plan_parser)
    plan_parser.set_defaults(func=cmd_plan)
    preflight_parser = sub.add_parser("preflight")
    _add_hydra_arguments(preflight_parser)
    preflight_parser.set_defaults(func=cmd_preflight)
    verify_parser = sub.add_parser("verify-manifest")
    verify_parser.add_argument("manifest")
    verify_parser.add_argument("--root")
    verify_parser.add_argument("--output")
    verify_parser.add_argument("--workers", type=int, default=1)
    verify_parser.set_defaults(func=cmd_verify_manifest)
    reports_parser = sub.add_parser("summarize-reports")
    reports_parser.add_argument("report_root")
    reports_parser.add_argument("--output")
    reports_parser.set_defaults(func=cmd_summarize_reports)
    zenodo_parser = sub.add_parser("zenodo-manifest")
    zenodo_parser.add_argument("record_id")
    zenodo_parser.add_argument("--output", required=True)
    zenodo_parser.set_defaults(func=cmd_zenodo_manifest)
    download_parser = sub.add_parser("download")
    download_parser.add_argument("manifest")
    download_parser.add_argument("--destination", required=True)
    download_parser.set_defaults(func=cmd_download)
    inventory_parser = sub.add_parser("inventory")
    inventory_parser.add_argument("paths", nargs="+")
    inventory_parser.add_argument("--output", required=True)
    inventory_parser.set_defaults(func=cmd_inventory)
    bucket_parser = sub.add_parser("bucket-index")
    bucket_parser.add_argument("paths", nargs="*")
    bucket_parser.add_argument("--root", required=True)
    bucket_parser.add_argument("--output", required=True)
    bucket_parser.add_argument("--buckets", type=int, default=256)
    bucket_parser.add_argument("--sha256", action="store_true")
    bucket_parser.add_argument("--pattern", action="append", default=[])
    bucket_parser.set_defaults(func=cmd_bucket_index)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
