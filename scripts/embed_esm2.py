#!/usr/bin/env python3
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

from polarfunc.esm_runner import completed_shard_is_valid, require_cuda, shard_for, validate_embeddings
from polarfunc.sequences import iter_fasta


def mock_embeddings(identifiers: list[str], dimension: int = 8) -> np.ndarray:
    rows = []
    for identifier in identifiers:
        seed = int.from_bytes(__import__("hashlib").sha256(identifier.encode()).digest()[:8], "little")
        rows.append(np.random.default_rng(seed).standard_normal(dimension).astype("float32"))
    return np.stack(rows) if rows else np.empty((0, dimension), dtype="float32")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def esm2_embeddings(
    identifiers: list[str],
    sequences: list[str],
    checkpoint: str,
    revision: str,
    device: str,
    batch_size: int,
    mixed_precision: bool,
    context_policy: str,
    local_files_only: bool,
) -> tuple[list[str], np.ndarray, dict[str, object], list[dict[str, object]]]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    if device.startswith("cuda"):
        require_cuda(torch.cuda.is_available())
    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, revision=revision, local_files_only=local_files_only)
    model = AutoModel.from_pretrained(
        checkpoint,
        revision=revision,
        local_files_only=local_files_only,
        add_pooling_layer=False,
    ).to(device).eval()
    model_load_seconds = time.perf_counter() - load_started
    context_candidates = [getattr(model.config, "max_position_embeddings", None), getattr(tokenizer, "model_max_length", None)]
    context = min(int(value) for value in context_candidates if value and int(value) < 10**8)
    eligible = [len(sequence) + 2 <= context for sequence in sequences]
    excluded = [
        {"CDHit_ID": identifier, "protein_length": len(sequence), "reason": f"length_plus_special_tokens_exceeds_{context}"}
        for identifier, sequence, keep in zip(identifiers, sequences, eligible)
        if not keep
    ]
    if excluded and context_policy == "error":
        raise ValueError(f"{len(excluded)} proteins exceed dynamically detected context length {context}")
    identifiers = [identifier for identifier, keep in zip(identifiers, eligible) if keep]
    sequences = [sequence for sequence, keep in zip(sequences, eligible) if keep]
    if not sequences:
        raise ValueError("No context-eligible proteins remain in this shard")
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    results = []
    effective_batch_sizes = []
    inference_started = time.perf_counter()
    with torch.inference_mode():
        offset = 0
        current_batch_size = batch_size
        while offset < len(sequences):
            batch = sequences[offset : offset + current_batch_size]
            try:
                tokens = tokenizer(batch, return_tensors="pt", padding=True, return_special_tokens_mask=True)
                special = tokens.pop("special_tokens_mask").to(device)
                tokens = {key: value.to(device) for key, value in tokens.items()}
                autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=mixed_precision and device.startswith("cuda"))
                with autocast:
                    hidden = model(**tokens).last_hidden_state
                mask = tokens["attention_mask"].bool() & ~special.bool()
                pooled = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
                results.append(pooled.float().cpu().numpy())
                effective_batch_sizes.append(len(batch))
                offset += len(batch)
            except torch.cuda.OutOfMemoryError:
                if current_batch_size == 1:
                    raise
                current_batch_size = max(1, current_batch_size // 2)
                torch.cuda.empty_cache()
    inference_seconds = time.perf_counter() - inference_started
    embeddings = np.concatenate(results)
    residues = sum(map(len, sequences))
    peak_vram = int(torch.cuda.max_memory_allocated() / 2**20) if device.startswith("cuda") else 0
    gpu_properties = torch.cuda.get_device_properties(torch.cuda.current_device()) if device.startswith("cuda") else None
    return identifiers, embeddings, {
        "checkpoint": checkpoint,
        "revision": revision,
        "context_length": context,
        "context_policy": context_policy,
        "embedding_dimension": int(embeddings.shape[1]),
        "pooling": "mean_valid_residue_tokens",
        "dtype": "bfloat16_autocast_float32_output" if mixed_precision else "float32",
        "mixed_precision": mixed_precision,
        "model_load_seconds": model_load_seconds,
        "inference_seconds": inference_seconds,
        "sequences_per_second": len(sequences) / inference_seconds,
        "residues": residues,
        "residues_per_second": residues / inference_seconds,
        "peak_gpu_memory_mib": peak_vram,
        "gpu_name": gpu_properties.name if gpu_properties else None,
        "gpu_total_memory_mib": int(gpu_properties.total_memory / 2**20) if gpu_properties else None,
        "gpu_uuid": str(getattr(gpu_properties, "uuid", "unknown")) if gpu_properties else None,
        "minimum_effective_batch_size": min(effective_batch_sizes),
        "maximum_effective_batch_size": max(effective_batch_sizes),
    }, excluded


def main() -> int:
    parser = argparse.ArgumentParser(description="Shard-aware resumable ESM2 embedding runner.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--throughput-limit", type=int)
    parser.add_argument("--backend", choices=["esm2", "mock"], default="esm2")
    parser.add_argument("--model-checkpoint", default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--context-policy", choices=["error", "exclude"], default="error")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("shard-id must be in [0, num-shards)")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.resume and completed_shard_is_valid(args.output_dir, args.shard_id):
        print(json.dumps({"status": "skipped_valid_completed_shard", "shard_id": args.shard_id}))
        return 0
    manifest = pd.read_parquet(args.manifest, columns=["CDHit_ID"])
    if not manifest.CDHit_ID.is_unique:
        raise ValueError("Manifest CDHit_ID values are not unique")
    selected = manifest.loc[manifest.CDHit_ID.astype(str).map(lambda value: shard_for(value, args.num_shards) == args.shard_id), "CDHit_ID"].astype(str).tolist()
    if args.throughput_limit is not None:
        selected = selected[: args.throughput_limit]
    plan = {"shard_id": args.shard_id, "num_shards": args.num_shards, "sequences": len(selected), "checkpoint": args.model_checkpoint, "revision": args.model_revision, "backend": args.backend}
    if args.dry_run:
        path = args.output_dir / f"shard_{args.shard_id:05d}.plan.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)
        print(json.dumps(plan, indent=2))
        return 0
    selected_set = set(selected)
    sequence_map = {identifier: sequence for identifier, sequence in iter_fasta(args.fasta) if identifier in selected_set}
    if set(sequence_map) != selected_set:
        raise RuntimeError(f"FASTA is missing {len(selected_set-set(sequence_map))} selected IDs")
    ordered = sorted(selected, key=lambda identifier: (len(sequence_map[identifier]), identifier))
    sequences = [sequence_map[identifier] for identifier in ordered]
    excluded: list[dict[str, object]] = []
    if args.backend == "mock":
        embeddings = mock_embeddings(ordered)
        model_meta = {"checkpoint": "mock", "revision": args.model_revision, "context_length": None, "context_policy": "not_applicable", "embedding_dimension": 8, "mixed_precision": False}
    else:
        ordered, embeddings, model_meta, excluded = esm2_embeddings(
            ordered,
            sequences,
            args.model_checkpoint,
            args.model_revision,
            args.device,
            args.batch_size,
            args.mixed_precision,
            args.context_policy,
            args.local_files_only,
        )
    validate_embeddings(ordered, embeddings, int(model_meta["embedding_dimension"]))
    norms = np.linalg.norm(embeddings, axis=1)
    if np.any(norms == 0):
        raise ValueError("Embedding matrix contains zero-norm rows")
    stem = f"shard_{args.shard_id:05d}"
    archive = args.output_dir / f"{stem}.npz"; metadata = args.output_dir / f"{stem}.json"; done = args.output_dir / f"{stem}.DONE"; checksum = args.output_dir / f"{stem}.sha256"; excluded_path = args.output_dir / f"{stem}.excluded.parquet"
    for path in (archive, metadata, done, checksum, excluded_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite incomplete/non-resumed shard: {path}")
    archive_tmp = args.output_dir / f".{stem}.npz.tmp"
    with archive_tmp.open("wb") as handle:
        np.savez_compressed(handle, ids=np.asarray(ordered), embeddings=embeddings.astype("float32"))
    archive_tmp.replace(archive)
    pd.DataFrame(excluded, columns=["CDHit_ID", "protein_length", "reason"]).to_parquet(excluded_path, index=False)
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    meta = {
        **plan,
        **model_meta,
        "sequences": len(ordered),
        "excluded_sequences": len(excluded),
        "device": args.device,
        "batch_size": args.batch_size,
        "hostname": socket.getfqdn(),
        "oar_job_id": os.environ.get("OAR_JOB_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "git_commit": commit,
        "manifest_sha256": sha256_file(args.manifest),
        "embedding_norm_min": float(norms.min()),
        "embedding_norm_mean": float(norms.mean()),
        "embedding_norm_max": float(norms.max()),
    }
    metadata_tmp = args.output_dir / f".{stem}.json.tmp"; metadata_tmp.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n"); metadata_tmp.replace(metadata)
    checksum.write_text(f"{sha256_file(archive)}  {archive.name}\n{sha256_file(metadata)}  {metadata.name}\n{sha256_file(excluded_path)}  {excluded_path.name}\n")
    if not completed_shard_is_valid(args.output_dir, args.shard_id):
        done.write_text("complete\n")
        if not completed_shard_is_valid(args.output_dir, args.shard_id):
            raise RuntimeError("Atomic shard validation failed")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
