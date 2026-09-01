#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the two-H100 OOD runtime")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--minimum-vram-gib", type=float, default=80.0)
    args = parser.parse_args()

    torch = importlib.import_module("torch")
    importlib.import_module("pytorch_ood")
    importlib.import_module("deel.puncc")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if torch.cuda.device_count() != 2:
        raise RuntimeError(
            f"exactly two visible GPUs are required; found {torch.cuda.device_count()}"
        )

    devices = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        total_gib = properties.total_memory / 2**30
        if total_gib < args.minimum_vram_gib:
            raise RuntimeError(
                f"GPU {index} has {total_gib:.2f} GiB; {args.minimum_vram_gib:.2f} required"
            )
        devices.append(
            {
                "index": index,
                "name": properties.name,
                "total_vram_gib": total_gib,
                "compute_capability": f"{properties.major}.{properties.minor}",
            }
        )

    report = {
        "status": "PASS",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "visible_gpu_count": len(devices),
        "devices": devices,
        "imports": {"pytorch_ood": "PASS", "deel.puncc": "PASS"},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
