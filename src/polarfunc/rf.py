from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from .schema import open_text


class RFSchemaError(ValueError):
    pass


@dataclass(frozen=True)
class RFInspection:
    path: str
    header_width: int
    data_widths: list[int]
    missing_identifier_header: bool
    columns: list[str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def inspect_rf_schema(path: Path, sample_rows: int = 20) -> RFInspection:
    path = Path(path)
    with open_text(path) as handle:
        header = next(csv.reader([handle.readline().rstrip("\r\n")], delimiter="\t"))
        widths: list[int] = []
        for index, row in enumerate(csv.reader(handle, delimiter="\t")):
            if index >= sample_rows:
                break
            widths.append(len(row))
    if not widths:
        raise RFSchemaError(f"RF file has no data rows: {path}")
    missing = all(width == len(header) + 1 for width in widths)
    if not missing and not all(width == len(header) for width in widths):
        raise RFSchemaError(f"Inconsistent RF widths: header={len(header)}, data={sorted(set(widths))}")
    columns = (["AGC_ID"] + header) if missing else (["AGC_ID"] + header[1:] if header[0] == "" else header)
    return RFInspection(str(path), len(columns), widths, missing, columns)


def repair_rf_table(path: Path, *, nrows: int | None = None) -> pd.DataFrame:
    inspection = inspect_rf_schema(path)
    frame = pd.read_csv(
        path,
        sep="\t",
        compression="infer",
        header=None,
        names=inspection.columns,
        skiprows=1,
        nrows=nrows,
        na_values=["NA", "NaN", ""],
        keep_default_na=True,
        low_memory=False,
    )
    if "AGC_ID" not in frame.columns:
        raise RFSchemaError("RF repair did not produce AGC_ID")
    for column in frame.columns:
        if column != "AGC_ID":
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    validate_rf_table(frame)
    return frame


def validate_rf_table(frame: pd.DataFrame) -> dict[str, object]:
    if "AGC_ID" not in frame.columns:
        raise RFSchemaError("Missing AGC_ID")
    duplicates = int(frame["AGC_ID"].duplicated().sum())
    if duplicates:
        raise RFSchemaError(f"RF table contains {duplicates} duplicate AGC_ID values")
    required = [column for column in ["mtry", "Rsquared_rf", "R2_caret", "MSE"] if column in frame.columns]
    if not {"Rsquared_rf", "R2_caret", "MSE"}.issubset(frame.columns):
        raise RFSchemaError("Missing required RF metric columns")
    nonnumeric = [column for column in required if not pd.api.types.is_numeric_dtype(frame[column])]
    if nonnumeric:
        raise RFSchemaError(f"Nonnumeric RF columns: {nonnumeric}")
    return {"passed": True, "rows": len(frame), "columns": len(frame.columns), "duplicate_agc_ids": duplicates}
