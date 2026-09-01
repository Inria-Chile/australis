from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
from typing import IO, Any

FASTA_SUFFIXES = {".fa", ".fasta", ".fna", ".faa"}
TABULAR_SUFFIXES = {".csv", ".tsv", ".txt"}


def logical_suffix(path: Path) -> str:
    suffixes = path.suffixes
    return (
        suffixes[-2].lower() if suffixes and suffixes[-1].lower() == ".gz" else path.suffix.lower()
    )


def open_text(path: Path) -> IO[str]:
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="")
    return path.open("r", encoding="utf-8", errors="replace", newline="")


def inspect_fasta(path: Path, preview_characters: int = 100) -> dict[str, Any]:
    sequence_count = 0
    residue_count = 0
    preview_header: str | None = None
    preview_parts: list[str] = []
    with open_text(path) as handle:
        for line in handle:
            if line.startswith(">"):
                sequence_count += 1
                if preview_header is None:
                    preview_header = line.rstrip("\r\n")
                continue
            sequence = line.strip()
            residue_count += len(sequence)
            if sequence_count == 1 and sum(map(len, preview_parts)) < preview_characters:
                preview_parts.append(sequence)
    return {
        "path": str(path),
        "kind": "fasta",
        "size_bytes": path.stat().st_size,
        "sequence_count": sequence_count,
        "residue_count": residue_count,
        "preview_header": preview_header,
        "preview_sequence": "".join(preview_parts)[:preview_characters],
    }


def inspect_tabular(path: Path, preview_rows: int = 3) -> dict[str, Any]:
    delimiter = "," if logical_suffix(path) == ".csv" else "\t"
    with open_text(path) as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        header = next(reader, [])
        preview: list[list[str]] = []
        row_count = 0
        malformed_rows = 0
        for row in reader:
            row_count += 1
            malformed_rows += int(len(row) != len(header))
            if len(preview) < preview_rows:
                preview.append(row)
    return {
        "path": str(path),
        "kind": "tabular",
        "size_bytes": path.stat().st_size,
        "rows": row_count,
        "columns": len(header),
        "malformed_rows": malformed_rows,
        "header": header,
        "preview_rows": preview,
    }


def inspect_path(path: Path) -> dict[str, Any]:
    suffix = logical_suffix(path)
    if suffix in FASTA_SUFFIXES:
        return inspect_fasta(path)
    if suffix in TABULAR_SUFFIXES:
        return inspect_tabular(path)
    return {"path": str(path), "kind": "binary", "size_bytes": path.stat().st_size}


def write_inventory(paths: list[Path], destination: Path) -> dict[str, Any]:
    records = [inspect_path(path) for path in paths]
    result = {"status": "PASS", "file_count": len(records), "files": records}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
