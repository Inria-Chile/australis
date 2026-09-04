#!/usr/bin/env python3
"""Deterministic, length-weighted ESM2 embeddings for proteins beyond model context."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from polarfunc.esm_runner import require_cuda, shard_for, validate_embeddings
from polarfunc.sequences import iter_fasta


POLICY = "nonoverlap_context_minus_special_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def windows(sequence: str, residues_per_window: int) -> list[str]:
    if not sequence:
        raise ValueError("Empty protein sequence")
    result = [sequence[start : start + residues_per_window] for start in range(0, len(sequence), residues_per_window)]
    if "".join(result) != sequence:
        raise RuntimeError("Chunking did not account for every residue")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--validate-single-window", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("Invalid shard")
    for suffix in ("npz", "json", "sha256", "DONE"):
        if (args.output_dir / f"shard_{args.shard_id:05d}.{suffix}").exists():
            raise FileExistsError("Refusing to overwrite chunked shard")
    manifest = pd.read_parquet(args.manifest, columns=["CDHit_ID"])
    selected = manifest.loc[
        manifest.CDHit_ID.astype(str).map(lambda value: shard_for(value, args.num_shards) == args.shard_id),
        "CDHit_ID",
    ].astype(str).tolist()
    selected_set = set(selected)
    sequence_map = {identifier: sequence for identifier, sequence in iter_fasta(args.fasta) if identifier in selected_set}
    if set(sequence_map) != selected_set:
        raise RuntimeError(f"FASTA mismatch: missing {len(selected_set - set(sequence_map))}")

    import torch
    from transformers import AutoModel, AutoTokenizer

    require_cuda(torch.cuda.is_available())
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint, revision=args.revision, local_files_only=args.local_files_only
    )
    model = AutoModel.from_pretrained(
        args.checkpoint,
        revision=args.revision,
        local_files_only=args.local_files_only,
        add_pooling_layer=False,
    ).to(args.device).eval()
    context = min(
        int(value)
        for value in (model.config.max_position_embeddings, tokenizer.model_max_length)
        if value and int(value) < 10**8
    )
    special_tokens = int(tokenizer.num_special_tokens_to_add(pair=False))
    residues_per_window = context - special_tokens
    if residues_per_window < 1:
        raise RuntimeError("Model context cannot contain residues")
    torch.cuda.reset_peak_memory_stats()

    def embed_batch(parts: list[str]) -> np.ndarray:
        tokens = tokenizer(parts, return_tensors="pt", padding=True, return_special_tokens_mask=True, truncation=False)
        special = tokens.pop("special_tokens_mask").to(args.device)
        tokens = {key: value.to(args.device) for key, value in tokens.items()}
        if int(tokens["input_ids"].shape[1]) > context:
            raise RuntimeError("Chunk exceeds model context")
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16,
            enabled=args.mixed_precision and args.device.startswith("cuda"),
        ):
            hidden = model(**tokens).last_hidden_state
        mask = tokens["attention_mask"].bool() & ~special.bool()
        return (
            (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        ).float().detach().cpu().numpy()

    ordered = sorted(selected, key=lambda value: (len(sequence_map[value]), value))
    embeddings: list[np.ndarray] = []
    total_windows = 0
    total_residues = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for identifier in ordered:
            sequence = sequence_map[identifier]
            parts = windows(sequence, residues_per_window)
            pooled = embed_batch(parts)
            lengths = np.asarray([len(part) for part in parts], dtype=np.float64)
            embedding = np.average(pooled.astype(np.float64), axis=0, weights=lengths).astype(np.float32)
            embeddings.append(embedding)
            total_windows += len(parts)
            total_residues += len(sequence)
    matrix = np.stack(embeddings)
    validate_embeddings(ordered, matrix, matrix.shape[1])
    if not np.isfinite(matrix).all() or np.any(np.linalg.norm(matrix, axis=1) == 0):
        raise RuntimeError("Invalid chunked embeddings")

    equivalence: dict[str, object] | None = None
    if args.validate_single_window:
        short = next((identifier for identifier in ordered if len(sequence_map[identifier]) <= residues_per_window), None)
        if short is None:
            raise RuntimeError("Single-window validation requested but probe has no context-safe protein")
        first = embed_batch([sequence_map[short]])[0]
        second = embed_batch(windows(sequence_map[short], residues_per_window))[0]
        max_error = float(np.max(np.abs(first - second)))
        equivalence = {"CDHit_ID": short, "max_absolute_error": max_error, "passed": max_error <= 1e-6}
        if not equivalence["passed"]:
            raise RuntimeError(f"Single-window equivalence failure: {equivalence}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"shard_{args.shard_id:05d}"
    archive = args.output_dir / f"{stem}.npz"
    temporary = args.output_dir / f".{stem}.npz.tmp"
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, ids=np.asarray(ordered), embeddings=matrix)
    temporary.replace(archive)
    elapsed = time.perf_counter() - started
    metadata = {
        "status": "PASS",
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "sequences": len(ordered),
        "windows": total_windows,
        "residues": total_residues,
        "embedding_dimension": int(matrix.shape[1]),
        "checkpoint": args.checkpoint,
        "revision": args.revision,
        "context_length": context,
        "special_tokens": special_tokens,
        "residues_per_window": residues_per_window,
        "chunking_policy": POLICY,
        "window_overlap": 0,
        "window_pooling": "mean_valid_residue_tokens",
        "window_aggregation": "residue_length_weighted_mean",
        "silent_truncation": False,
        "single_window_equivalence": equivalence,
        "dtype": "bfloat16_autocast_float32_output" if args.mixed_precision else "float32",
        "inference_seconds": elapsed,
        "peak_gpu_memory_mib": int(torch.cuda.max_memory_allocated() / 2**20),
        "host": socket.getfqdn(),
        "oar_job_id": os.environ.get("OAR_JOB_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "git_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "manifest_sha256": sha256(args.manifest),
    }
    metadata_path = args.output_dir / f"{stem}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    checksum_path = args.output_dir / f"{stem}.sha256"
    checksum_path.write_text(f"{sha256(archive)}  {archive.name}\n{sha256(metadata_path)}  {metadata_path.name}\n")
    (args.output_dir / f"{stem}.DONE").write_text("PASS\n")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
