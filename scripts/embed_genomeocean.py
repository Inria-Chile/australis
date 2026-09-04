#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from polarfunc.esm_runner import completed_shard_is_valid, require_cuda, shard_for, validate_embeddings
from polarfunc.genomeocean import pool_hidden
from polarfunc.sequences import iter_fasta


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Resumable GenomeOcean inference runner.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--pooling", choices=["mean", "last-valid-token"], default="mean")
    parser.add_argument("--context-policy", choices=["error", "exclude"], default="error")
    parser.add_argument("--attn-implementation", choices=["sdpa", "eager", "flash_attention_2"], default="sdpa")
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--throughput-limit", type=int)
    parser.add_argument("--backend", choices=["genomeocean", "mock"], default="genomeocean")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.resume and completed_shard_is_valid(args.output_dir, args.shard_id):
        print(json.dumps({"status": "skipped_valid_completed_shard", "shard_id": args.shard_id}))
        return 0

    manifest = pd.read_parquet(args.manifest, columns=["CDHit_ID"])
    selected = manifest.loc[
        manifest.CDHit_ID.astype(str).map(lambda value: shard_for(value, args.num_shards) == args.shard_id),
        "CDHit_ID",
    ].astype(str).tolist()
    if args.throughput_limit is not None:
        selected = selected[: args.throughput_limit]
    plan = {
        "expected_sequences": len(selected), "shard_id": args.shard_id,
        "num_shards": args.num_shards, "checkpoint": args.checkpoint,
        "revision": args.revision, "pooling": args.pooling, "backend": args.backend,
        "context_policy": args.context_policy, "attn_implementation": args.attn_implementation,
    }
    if args.dry_run:
        path = args.output_dir / f"shard_{args.shard_id:05d}.plan.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
        tmp.replace(path)
        print(json.dumps(plan, indent=2))
        return 0

    selected_set = set(selected)
    seqmap = {identifier: sequence for identifier, sequence in iter_fasta(args.fasta) if identifier in selected_set}
    if set(seqmap) != selected_set:
        raise RuntimeError("GenomeOcean FASTA/manifest mismatch")
    ordered = sorted(selected, key=lambda value: (len(seqmap[value]), value))
    started = time.perf_counter()
    excluded_rows: list[dict[str, object]] = []

    if args.backend == "mock":
        embeddings = np.stack([np.arange(8, dtype="float32") for _ in ordered])
        context = None
        peak_gpu_memory_mib = 0
    else:
        import torch
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        if args.device.startswith("cuda"):
            require_cuda(torch.cuda.is_available())
            torch.cuda.reset_peak_memory_stats()
        tokenizer = AutoTokenizer.from_pretrained(
            args.checkpoint, revision=args.revision, trust_remote_code=False,
            local_files_only=args.local_files_only, padding_side="left",
        )
        config = AutoConfig.from_pretrained(
            args.checkpoint, revision=args.revision, trust_remote_code=False,
            local_files_only=args.local_files_only,
        )
        config.use_cache = False
        dtype = torch.bfloat16 if args.mixed_precision else torch.float32
        model = AutoModel.from_pretrained(
            args.checkpoint, revision=args.revision, config=config, trust_remote_code=False,
            local_files_only=args.local_files_only, torch_dtype=dtype,
            attn_implementation=args.attn_implementation,
        ).to(args.device).eval()
        context_values = [
            int(value) for value in (
                getattr(model.config, "max_position_embeddings", None),
                getattr(tokenizer, "model_max_length", None),
            ) if value and int(value) < 10**8
        ]
        context = min(context_values)
        encoded_lengths = tokenizer(
            [seqmap[value] for value in ordered], add_special_tokens=True, padding=False,
            truncation=False, return_length=True,
        )["length"]
        too_long = [value for value, length in zip(ordered, encoded_lengths) if int(length) > context]
        if too_long and args.context_policy == "error":
            raise RuntimeError(f"{len(too_long)} sequences exceed GenomeOcean context {context}")
        excluded_rows = [
            {"CDHit_ID": value, "reason": "token_length_exceeds_context", "context_length": context}
            for value in too_long
        ]
        excluded_set = set(too_long)
        ordered = [value for value in ordered if value not in excluded_set]
        rows = []
        with torch.inference_mode():
            for offset in range(0, len(ordered), args.batch_size):
                batch = [seqmap[value] for value in ordered[offset : offset + args.batch_size]]
                tokens = tokenizer(
                    batch, return_tensors="pt", padding=True, truncation=False,
                    return_special_tokens_mask=True,
                )
                special = tokens.pop("special_tokens_mask").numpy()
                attention = tokens["attention_mask"].numpy()
                device_tokens = {key: value.to(args.device, non_blocking=True) for key, value in tokens.items()}
                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16,
                    enabled=args.mixed_precision and args.device.startswith("cuda"),
                ):
                    output = model(**device_tokens, use_cache=False).last_hidden_state
                hidden = output.float().cpu().numpy()
                rows.append(pool_hidden(hidden, attention.astype(bool) & ~special.astype(bool), args.pooling))
        embeddings = np.concatenate(rows) if rows else np.empty((0, int(model.config.hidden_size)), dtype=np.float32)
        peak_gpu_memory_mib = int(torch.cuda.max_memory_allocated() / 2**20) if args.device.startswith("cuda") else 0

    validate_embeddings(ordered, embeddings)
    elapsed = time.perf_counter() - started
    stem = f"shard_{args.shard_id:05d}"
    archive = args.output_dir / f"{stem}.npz"
    metadata = args.output_dir / f"{stem}.json"
    excluded_path = args.output_dir / f"{stem}.excluded.parquet"
    checksum_path = args.output_dir / f"{stem}.sha256"
    done = args.output_dir / f"{stem}.DONE"
    archive_tmp = args.output_dir / f".{stem}.npz.tmp"
    with archive_tmp.open("wb") as handle:
        np.savez_compressed(handle, ids=np.asarray(ordered), embeddings=embeddings.astype("float32"))
    archive_tmp.replace(archive)
    pd.DataFrame(excluded_rows, columns=["CDHit_ID", "reason", "context_length"]).to_parquet(excluded_path, index=False)
    payload = {
        **plan, "sequences": len(ordered), "excluded_context": len(excluded_rows),
        "embedding_dimension": int(embeddings.shape[1]), "context_length": context,
        "dtype": "bfloat16_autocast_float32_output" if args.mixed_precision else "float32",
        "inference_seconds": elapsed, "sequences_per_second": len(ordered) / elapsed if elapsed else None,
        "peak_gpu_memory_mib": peak_gpu_memory_mib,
    }
    metadata_tmp = args.output_dir / f".{stem}.json.tmp"
    metadata_tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    metadata_tmp.replace(metadata)
    checksum_path.write_text("".join(
        f"{file_sha256(path)}  {path.name}\n" for path in (archive, metadata, excluded_path)
    ))
    done.write_text("complete\n")
    if not completed_shard_is_valid(args.output_dir, args.shard_id):
        raise RuntimeError("GenomeOcean shard validation failed")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
