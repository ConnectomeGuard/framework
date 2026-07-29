from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


VALID_DIAGNOSES = {"ASD", "CONTROL"}
VALID_SPLITS = {"train", "validation", "site_heldout_test", "unassigned"}
VALID_QUALITY_STATUSES = {"pass", "fail", "unknown"}


@dataclass
class SubjectRecord:
    subject_id: str
    dataset: str
    diagnosis: str
    age: float
    sex: str
    site: str
    scanner: str
    modality: str
    t1_path: Optional[str] = field(default=None)
    dwi_path: Optional[str] = field(default=None)
    func_path: Optional[str] = field(default=None)
    phenotype_path: Optional[str] = field(default=None)
    quality_status: str = field(default="unknown")
    split: str = field(default="unassigned")

    def has_any_modality_path(self) -> bool:
        return any(p is not None and str(p).strip() != "" for p in [self.t1_path, self.dwi_path, self.func_path])

    def to_dict(self) -> dict:
        return {
            "subject_id": self.subject_id,
            "dataset": self.dataset,
            "diagnosis": self.diagnosis,
            "age": self.age,
            "sex": self.sex,
            "site": self.site,
            "scanner": self.scanner,
            "modality": self.modality,
            "t1_path": self.t1_path,
            "dwi_path": self.dwi_path,
            "func_path": self.func_path,
            "phenotype_path": self.phenotype_path,
            "quality_status": self.quality_status,
            "split": self.split,
        }
