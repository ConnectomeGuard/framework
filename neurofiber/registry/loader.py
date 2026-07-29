from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from neurofiber.data.subject_record import SubjectRecord


_REQUIRED_COLUMNS = [
    "subject_id", "dataset", "diagnosis", "age",
    "sex", "site", "scanner", "modality",
]

_OPTIONAL_COLUMNS = [
    "t1_path", "dwi_path", "func_path", "phenotype_path",
    "quality_status", "split",
]

_DEFAULTS = {
    "t1_path": None,
    "dwi_path": None,
    "func_path": None,
    "phenotype_path": None,
    "quality_status": "unknown",
    "split": "unassigned",
}


def _nan_to_none(value: object) -> object:
    if value is None:
        return None
    try:
        if math.isnan(float(value)):  # type: ignore[arg-type]
            return None
    except (TypeError, ValueError):
        pass
    return value


def load_dataset_index(path: str | Path) -> list[SubjectRecord]:
    df = pd.read_csv(Path(path), dtype=str)

    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    for col, default in _DEFAULTS.items():
        if col not in df.columns:
            df[col] = default

    records: list[SubjectRecord] = []
    for _, row in df.iterrows():
        records.append(
            SubjectRecord(
                subject_id=str(row["subject_id"]).strip(),
                dataset=str(row["dataset"]).strip(),
                diagnosis=str(row["diagnosis"]).strip(),
                age=float(row["age"]),
                sex=str(row["sex"]).strip(),
                site=str(row["site"]).strip(),
                scanner=str(row["scanner"]).strip(),
                modality=str(row["modality"]).strip(),
                t1_path=_nan_to_none(row.get("t1_path")),          # type: ignore[arg-type]
                dwi_path=_nan_to_none(row.get("dwi_path")),        # type: ignore[arg-type]
                func_path=_nan_to_none(row.get("func_path")),      # type: ignore[arg-type]
                phenotype_path=_nan_to_none(row.get("phenotype_path")),  # type: ignore[arg-type]
                quality_status=str(row.get("quality_status", "unknown")).strip(),
                split=str(row.get("split", "unassigned")).strip(),
            )
        )
    return records


def save_dataset_index(records: list[SubjectRecord], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([r.to_dict() for r in records])
    df.to_csv(output, index=False)
