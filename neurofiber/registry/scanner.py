from __future__ import annotations

from pathlib import Path

from neurofiber.data.validation_record import ValidationRecord
from neurofiber.registry.dti_validator import validate_dti
from neurofiber.utils.logging import get_logger

logger = get_logger(__name__)

_SESSION = "session_1"


def scan_subjects(
    dataset_root: Path | str,
    site: str,
    dataset: str,
    session: str = _SESSION,
) -> list[ValidationRecord]:
    """Scan all subject directories under *dataset_root* and return one ValidationRecord each."""
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    subject_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    logger.info("Found %d subject directories in %s", len(subject_dirs), root)

    records: list[ValidationRecord] = []
    for subject_dir in subject_dirs:
        rec = _scan_one(subject_dir, site=site, dataset=dataset, session=session)
        records.append(rec)
        logger.debug("  %-30s  status=%s", rec.subject_id, rec.validation_status)

    return records


def _scan_one(
    subject_dir: Path,
    site: str,
    dataset: str,
    session: str,
) -> ValidationRecord:
    subject_id = subject_dir.name
    session_dir = subject_dir / session

    anat_path = session_dir / "anat_1" / "anat.nii.gz"
    dti_path  = session_dir / "dti_1"  / "dti.nii.gz"
    bval_path = session_dir / "dti_1"  / "dti.bval"
    bvec_path = session_dir / "dti_1"  / "dti.bvec"
    rest_path = session_dir / "rest_1" / "rest.nii.gz"

    rec = ValidationRecord(
        subject_id=subject_id,
        site=site,
        dataset=dataset,
        anat_exists=anat_path.exists(),
        dti_exists=dti_path.exists(),
        bval_exists=bval_path.exists(),
        bvec_exists=bvec_path.exists(),
        rest_exists=rest_path.exists(),
    )

    missing = [
        name for name, flag in [
            ("anat", rec.anat_exists),
            ("dti",  rec.dti_exists),
            ("bval", rec.bval_exists),
            ("bvec", rec.bvec_exists),
            ("rest", rec.rest_exists),
        ] if not flag
    ]

    if missing:
        rec.validation_status = "missing_files"
        logger.warning("  [%s] missing: %s", subject_id, ", ".join(missing))
        return rec

    # All files present — run DTI consistency check
    dti = validate_dti(dti_path, bval_path, bvec_path)
    rec.dti_volume_count = dti.volume_count
    rec.bval_count       = dti.bval_count
    rec.bvec_count       = dti.bvec_count
    rec.validation_status = dti.status

    if dti.errors:
        for err in dti.errors:
            logger.warning("  [%s] %s", subject_id, err)

    return rec
