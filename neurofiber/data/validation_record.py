from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ValidationRecord:
    subject_id: str
    site: str
    dataset: str
    anat_exists: bool
    dti_exists: bool
    bval_exists: bool
    bvec_exists: bool
    rest_exists: bool
    dti_volume_count: Optional[int] = field(default=None)
    bval_count: Optional[int] = field(default=None)
    bvec_count: Optional[int] = field(default=None)
    validation_status: str = field(default="unknown")

    def all_files_present(self) -> bool:
        return all([
            self.anat_exists,
            self.dti_exists,
            self.bval_exists,
            self.bvec_exists,
            self.rest_exists,
        ])

    def dti_consistent(self) -> bool:
        return (
            self.dti_volume_count is not None
            and self.bval_count is not None
            and self.bvec_count is not None
            and self.dti_volume_count == self.bval_count == self.bvec_count
        )

    def to_dict(self) -> dict:
        return {
            "subject_id": self.subject_id,
            "site": self.site,
            "dataset": self.dataset,
            "anat_exists": self.anat_exists,
            "dti_exists": self.dti_exists,
            "bval_exists": self.bval_exists,
            "bvec_exists": self.bvec_exists,
            "rest_exists": self.rest_exists,
            "dti_volume_count": self.dti_volume_count,
            "bval_count": self.bval_count,
            "bvec_count": self.bvec_count,
            "validation_status": self.validation_status,
        }
