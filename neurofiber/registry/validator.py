from __future__ import annotations

from neurofiber.data.subject_record import SubjectRecord, VALID_DIAGNOSES


class ValidationError(Exception):
    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__(f"{len(errors)} validation error(s):\n" + "\n".join(f"  • {e}" for e in errors))


def validate_records(records: list[SubjectRecord], raise_on_error: bool = True) -> list[str]:
    errors: list[str] = []

    _check_required_fields(records, errors)
    _check_diagnoses(records, errors)
    _check_ages(records, errors)
    _check_unique_ids(records, errors)
    _check_modality_paths(records, errors)

    if raise_on_error and errors:
        raise ValidationError(errors)

    return errors


def _check_required_fields(records: list[SubjectRecord], errors: list[str]) -> None:
    required = ["subject_id", "dataset", "diagnosis", "sex", "site", "scanner", "modality"]
    for r in records:
        for f in required:
            val = getattr(r, f, None)
            if val is None or str(val).strip() == "":
                errors.append(f"[{r.subject_id}] missing required field: {f}")


def _check_diagnoses(records: list[SubjectRecord], errors: list[str]) -> None:
    for r in records:
        if r.diagnosis not in VALID_DIAGNOSES:
            errors.append(
                f"[{r.subject_id}] invalid diagnosis '{r.diagnosis}' — must be one of {sorted(VALID_DIAGNOSES)}"
            )


def _check_ages(records: list[SubjectRecord], errors: list[str]) -> None:
    for r in records:
        try:
            age = float(r.age)
            if age < 0 or age > 150:
                errors.append(f"[{r.subject_id}] age {age} is outside plausible range [0, 150]")
        except (TypeError, ValueError):
            errors.append(f"[{r.subject_id}] age must be numeric, got '{r.age}'")


def _check_unique_ids(records: list[SubjectRecord], errors: list[str]) -> None:
    seen: dict[str, int] = {}
    for r in records:
        seen[r.subject_id] = seen.get(r.subject_id, 0) + 1
    for sid, count in seen.items():
        if count > 1:
            errors.append(f"duplicate subject_id '{sid}' appears {count} times")


def _check_modality_paths(records: list[SubjectRecord], errors: list[str]) -> None:
    for r in records:
        if not r.has_any_modality_path():
            errors.append(f"[{r.subject_id}] no modality path provided (t1_path, dwi_path, or func_path required)")
