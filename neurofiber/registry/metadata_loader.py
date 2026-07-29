"""
ABIDE II phenotypic metadata loader.

Normalizes the raw ABIDE II CSV fields to project-standard names and values:
  SUB_ID        → subject_id (str)
  SITE_ID       → site (str, e.g. "ETH")
  DX_GROUP      → diagnosis  (1→"ASD", 2→"CONTROL")
  AGE_AT_SCAN   → age        (float)
  SEX           → sex        (1→"M", 2→"F")
  FIQ           → fiq        (float or None)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from neurofiber.utils.logging import get_logger

logger = get_logger(__name__)

_DX_MAP  = {1: "ASD", 2: "CONTROL"}
_SEX_MAP = {1: "M",   2: "F"}


@dataclass
class SubjectMetadata:
    subject_id: str
    site: str
    dataset: str
    diagnosis: str
    age: float
    sex: str
    fiq: Optional[float]

    def to_dict(self) -> dict:
        return {
            "subject_id": self.subject_id,
            "site":       self.site,
            "dataset":    self.dataset,
            "diagnosis":  self.diagnosis,
            "age":        self.age,
            "sex":        self.sex,
            "fiq":        self.fiq,
        }


_CANDIDATE_ENCODINGS = ("utf-8", "latin-1", "cp1252", "iso-8859-1")


def _read_csv_any_encoding(path: Path) -> pd.DataFrame:
    """Try common encodings in order; raise if all fail."""
    for enc in _CANDIDATE_ENCODINGS:
        try:
            df = pd.read_csv(path, encoding=enc)
            if enc != "utf-8":
                logger.info("CSV '%s' read with encoding %s (not UTF-8)", path.name, enc)
            return df
        except UnicodeDecodeError:
            continue
    raise ValueError(
        f"Cannot decode '{path}' with any of: {_CANDIDATE_ENCODINGS}"
    )


def load_abide2_metadata(
    csv_path: Path | str,
    site: str,
    dataset: str = "ABIDE_II",
) -> dict[str, SubjectMetadata]:
    """
    Load and normalise an ABIDE II phenotypic CSV.

    Automatically detects non-UTF-8 encodings (e.g. latin-1 used by
    some ABIDE II sites such as Georgetown).

    Returns a dict keyed by subject_id (str) for O(1) join.
    """
    df = _read_csv_any_encoding(Path(csv_path))
    df.columns = df.columns.str.strip()

    required = {"SUB_ID", "DX_GROUP", "AGE_AT_SCAN", "SEX"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"Metadata CSV missing columns: {missing_cols}")

    records: dict[str, SubjectMetadata] = {}
    for _, row in df.iterrows():
        subject_id = str(int(row["SUB_ID"]))
        dx_raw = int(row["DX_GROUP"])
        sex_raw = int(row["SEX"])
        fiq_raw = row.get("FIQ", None)

        diagnosis = _DX_MAP.get(dx_raw, f"UNKNOWN_{dx_raw}")
        sex = _SEX_MAP.get(sex_raw, f"UNKNOWN_{sex_raw}")
        fiq = float(fiq_raw) if pd.notna(fiq_raw) else None

        records[subject_id] = SubjectMetadata(
            subject_id=subject_id,
            site=site,
            dataset=dataset,
            diagnosis=diagnosis,
            age=float(str(row["AGE_AT_SCAN"]).strip()),
            sex=sex,
            fiq=fiq,
        )

    logger.info("Loaded %d metadata records from %s", len(records), csv_path)
    return records
