from __future__ import annotations

import csv
import gzip
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, TextIO


class SchemaError(ValueError):
    pass


@dataclass(frozen=True)
class DelimitedSchema:
    path: str
    delimiter: str
    columns: list[str]
    sample_widths: list[int]
    sample_rows: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def open_text(path: Path) -> TextIO:
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="")
    return path.open("r", encoding="utf-8", errors="replace", newline="")


def discover_delimited_schema(
    path: Path,
    *,
    delimiter: str | None = None,
    sample_rows: int = 20,
) -> DelimitedSchema:
    path = Path(path)
    with open_text(path) as handle:
        first = handle.readline()
        if not first:
            raise SchemaError(f"Empty delimited file: {path}")
        if delimiter is None:
            delimiter = "\t" if first.count("\t") >= first.count(",") else ","
        columns = next(csv.reader([first.rstrip("\r\n")], delimiter=delimiter))
        widths: list[int] = []
        reader = csv.reader(handle, delimiter=delimiter)
        for index, row in enumerate(reader):
            if index >= sample_rows:
                break
            widths.append(len(row))
    return DelimitedSchema(str(path), delimiter, columns, widths, len(widths))


def resolve_semantics(
    columns: Iterable[str],
    aliases: dict[str, list[str]],
    *,
    required: set[str] | None = None,
) -> dict[str, str | None]:
    available = set(columns)
    mapping = {semantic: next((name for name in names if name in available), None) for semantic, names in aliases.items()}
    missing = sorted(semantic for semantic in (required or set()) if mapping.get(semantic) is None)
    if missing:
        raise SchemaError(f"Missing required semantic fields: {', '.join(missing)}")
    return mapping

