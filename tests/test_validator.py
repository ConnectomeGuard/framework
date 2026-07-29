from __future__ import annotations

import pytest

from neurofiber.data.subject_record import SubjectRecord
from neurofiber.registry.validator import validate_records, ValidationError


def _make_record(**kwargs) -> SubjectRecord:
    defaults = dict(
        subject_id="sub-001",
        dataset="ABIDE",
        diagnosis="ASD",
        age=12.5,
        sex="M",
        site="NYU",
        scanner="Siemens",
        modality="T1w",
        t1_path="/data/sub-001/T1w.nii.gz",
    )
    defaults.update(kwargs)
    return SubjectRecord(**defaults)


def test_valid_record_produces_no_errors():
    records = [_make_record()]
    errors = validate_records(records, raise_on_error=False)
    assert errors == []


def test_invalid_diagnosis_caught():
    records = [_make_record(diagnosis="AUTISM")]
    errors = validate_records(records, raise_on_error=False)
    assert any("diagnosis" in e for e in errors)


def test_non_numeric_age_caught():
    r = _make_record()
    r.age = "not_a_number"  # type: ignore[assignment]
    errors = validate_records([r], raise_on_error=False)
    assert any("age" in e for e in errors)


def test_duplicate_subject_id_caught():
    records = [_make_record(), _make_record()]
    errors = validate_records(records, raise_on_error=False)
    assert any("duplicate" in e.lower() for e in errors)


def test_no_modality_path_caught():
    records = [_make_record(t1_path=None, dwi_path=None, func_path=None)]
    errors = validate_records(records, raise_on_error=False)
    assert any("modality path" in e for e in errors)


def test_missing_required_field_caught():
    r = _make_record()
    r.site = ""
    errors = validate_records([r], raise_on_error=False)
    assert any("site" in e for e in errors)


def test_raises_validation_error_when_requested():
    records = [_make_record(diagnosis="UNKNOWN")]
    with pytest.raises(ValidationError) as exc_info:
        validate_records(records, raise_on_error=True)
    assert exc_info.value.errors


def test_multiple_valid_records():
    records = [
        _make_record(subject_id="sub-001"),
        _make_record(subject_id="sub-002", diagnosis="CONTROL", dwi_path="/data/sub-002/dwi.nii.gz"),
    ]
    errors = validate_records(records, raise_on_error=False)
    assert errors == []
