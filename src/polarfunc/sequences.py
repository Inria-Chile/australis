from __future__ import annotations

from collections.abc import Iterable
import hashlib
from pathlib import Path

import pandas as pd


STANDARD_CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L", "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*", "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L", "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q", "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M", "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K", "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V", "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E", "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}
TABLE_11_START_CODONS = {"TTG", "CTG", "ATT", "ATC", "ATA", "ATG", "GTG"}


def sequence_qc(sequence: bytes) -> dict[str, object]:
    sequence = sequence.upper()
    length = len(sequence)
    a = sequence.count(b"A")
    c = sequence.count(b"C")
    g = sequence.count(b"G")
    t = sequence.count(b"T")
    n = sequence.count(b"N")
    canonical = a + c + g + t
    ambiguous = length - canonical - n
    return {
        "length_nt": length,
        "GC_fraction": (g + c) / canonical if canonical else float("nan"),
        "N_fraction": n / length if length else float("nan"),
        "ambiguous_fraction": ambiguous / length if length else float("nan"),
        "sequence_sha256": hashlib.sha256(sequence).hexdigest(),
    }


def extract_requested_fasta(lines: Iterable[bytes], requested: set[str], output: Path) -> tuple[pd.DataFrame, int]:
    output = Path(output)
    found: set[str] = set()
    rows: list[dict[str, object]] = []
    current_id: str | None = None
    chunks: list[bytes] = []
    scanned = 0

    def flush(handle) -> None:
        nonlocal current_id, chunks
        if current_id is None:
            return
        scanned_sequence = b"".join(chunks).replace(b" ", b"").upper()
        if current_id in requested:
            if current_id in found:
                raise ValueError(f"Duplicate requested FASTA header: {current_id}")
            found.add(current_id)
            handle.write(b">" + current_id.encode("utf-8") + b"\n" + scanned_sequence + b"\n")
            rows.append({"CDHit_ID": current_id, **sequence_qc(scanned_sequence)})
        current_id = None
        chunks = []

    with output.open("wb") as handle:
        for raw in lines:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(b">"):
                flush(handle)
                scanned += 1
                current_id = line[1:].split(None, 1)[0].decode("utf-8")
                chunks = []
            elif current_id in requested:
                chunks.append(line)
        flush(handle)
    return pd.DataFrame(rows), scanned


def translate_cds(sequence: str, *, recode_start_as_methionine: bool = True) -> tuple[str, dict[str, object]]:
    sequence = sequence.upper().replace(" ", "")
    complete_length = len(sequence) - len(sequence) % 3
    amino_acids: list[str] = []
    ambiguous_codons = 0
    for offset in range(0, complete_length, 3):
        codon = sequence[offset : offset + 3]
        amino_acid = STANDARD_CODON_TABLE.get(codon)
        if amino_acid is None:
            amino_acid = "X"
            ambiguous_codons += 1
        elif offset == 0 and recode_start_as_methionine and codon in TABLE_11_START_CODONS:
            amino_acid = "M"
        amino_acids.append(amino_acid)
    terminal_stop = bool(amino_acids and amino_acids[-1] == "*")
    internal_stop_count = sum(amino_acid == "*" for amino_acid in amino_acids[:-1] if terminal_stop) if terminal_stop else amino_acids.count("*")
    translated = "".join(amino_acids[:-1] if terminal_stop else amino_acids)
    qc = {
        "nt_length": len(sequence),
        "nt_length_mod3": len(sequence) % 3,
        "terminal_stop": terminal_stop,
        "internal_stop_count": internal_stop_count,
        "ambiguous_codon_count": ambiguous_codons,
        "protein_length": len(translated),
        "X_fraction": translated.count("X") / len(translated) if translated else float("nan"),
    }
    return translated, qc


def protein_is_valid(qc: dict[str, object], policy: dict[str, object]) -> bool:
    return bool(
        int(qc["protein_length"]) >= int(policy["minimum_protein_length"])
        and (bool(policy["allow_partial_final_codon"]) or int(qc["nt_length_mod3"]) == 0)
        and int(qc["internal_stop_count"]) <= int(policy["maximum_internal_stop_count"])
        and int(qc["ambiguous_codon_count"]) <= int(policy["maximum_ambiguous_codon_count"])
        and float(qc["X_fraction"]) <= float(policy["maximum_X_fraction"])
    )


def iter_fasta(path: Path) -> Iterable[tuple[str, str]]:
    identifier: str | None = None
    chunks: list[str] = []
    with Path(path).open() as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if identifier is not None:
                    yield identifier, "".join(chunks)
                identifier = line[1:].split(None, 1)[0]
                chunks = []
            else:
                if identifier is None:
                    raise ValueError("FASTA sequence encountered before first header")
                chunks.append(line)
    if identifier is not None:
        yield identifier, "".join(chunks)


def dataframe_to_markdown(frame: pd.DataFrame, *, float_digits: int = 6) -> str:
    def render(value: object) -> str:
        if isinstance(value, float):
            return f"{value:.{float_digits}f}"
        return str(value)

    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    separator = "| " + " | ".join("---" for _ in frame.columns) + " |"
    rows = ["| " + " | ".join(render(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *rows])
