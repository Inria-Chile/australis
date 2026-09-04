#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoConfig, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--report-md", type=Path, required=True)
    args = parser.parse_args()
    if args.report_json.exists() or args.report_md.exists():
        raise SystemExit("Refusing to overwrite context audit")

    identifiers = set(pd.read_parquet(args.manifest, columns=["CDHit_ID"]).CDHit_ID.astype(str))
    sequences = []
    current_id = None
    chunks = []
    with args.fasta.open() as handle:
        for line in handle:
            if line.startswith(">"):
                if current_id in identifiers:
                    sequences.append("".join(chunks))
                current_id = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line.strip())
        if current_id in identifiers:
            sequences.append("".join(chunks))
    if len(sequences) != len(identifiers):
        raise SystemExit(f"FASTA/manifest mismatch: {len(sequences)} vs {len(identifiers)}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint, revision=args.revision, local_files_only=True,
        trust_remote_code=False,
    )
    config = AutoConfig.from_pretrained(
        args.checkpoint, revision=args.revision, local_files_only=True,
        trust_remote_code=False,
    )
    context = min(
        int(value) for value in (config.max_position_embeddings, tokenizer.model_max_length)
        if value and int(value) < 10**8
    )
    lengths = np.asarray(tokenizer(
        sequences, add_special_tokens=True, padding=False, truncation=False,
        return_length=True,
    )["length"], dtype=np.int64)
    report = {
        "status": "PASS",
        "sequences": len(lengths),
        "context_length": context,
        "token_length": {
            "min": int(lengths.min()), "median": float(np.median(lengths)),
            "p90": float(np.quantile(lengths, 0.90)), "p95": float(np.quantile(lengths, 0.95)),
            "p99": float(np.quantile(lengths, 0.99)), "max": int(lengths.max()),
        },
        "exceeding_context": int((lengths > context).sum()),
        "fraction_exceeding_context": float((lengths > context).mean()),
        "revision": args.revision,
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2) + "\n")
    args.report_md.write_text(
        "# GenomeOcean DNA context audit\n\n"
        f"- Status: **PASS**\n- Sequences: {len(lengths):,}\n- Context: {context:,} tokens\n"
        f"- Median: {report['token_length']['median']:,.0f}\n- P99: {report['token_length']['p99']:,.0f}\n"
        f"- Maximum: {report['token_length']['max']:,}\n- Exceeding context: {report['exceeding_context']:,}\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
