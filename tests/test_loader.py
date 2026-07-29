from __future__ import annotations

import io
import textwrap
from pathlib import Path

import pandas as pd
import pytest

from neurofiber.registry.loader import load_dataset_index, save_dataset_index
from neurofiber.data.subject_record import SubjectRecord


MINIMAL_CSV = textwrap.dedent("""\
    subject_id,dataset,diagnosis,age,sex,site,scanner,modality,t1_path
    sub-001,ABIDE,ASD,12.5,M,NYU,Siemens,T1w,/data/sub-001/anat/T1w.nii.gz
    sub-002,ABIDE,CONTROL,14.0,F,NYU,Siemens,T1w,/data/sub-002/anat/T1w.nii.gz
""")


def _write_csv(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "index.csv"
    p.write_text(content)
    return p


def test_load_returns_subject_records(tmp_path):
    path = _write_csv(tmp_path, MINIMAL_CSV)
    records = load_dataset_index(path)
    assert len(records) == 2
    assert all(isinstance(r, SubjectRecord) for r in records)


def test_load_field_values(tmp_path):
    path = _write_csv(tmp_path, MINIMAL_CSV)
    r = load_dataset_index(path)[0]
    assert r.subject_id == "sub-001"
    assert r.diagnosis == "ASD"
    assert r.age == 12.5
    assert r.t1_path == "/data/sub-001/anat/T1w.nii.gz"
    assert r.dwi_path is None
    assert r.split == "unassigned"


def test_load_missing_required_column_raises(tmp_path):
    bad_csv = textwrap.dedent("""\
        subject_id,dataset,age
        sub-001,ABIDE,10
    """)
    path = _write_csv(tmp_path, bad_csv)
    with pytest.raises(ValueError, match="missing required columns"):
        load_dataset_index(path)


def test_roundtrip(tmp_path):
    path = _write_csv(tmp_path, MINIMAL_CSV)
    records = load_dataset_index(path)
    out = tmp_path / "out.csv"
    save_dataset_index(records, out)
    reloaded = load_dataset_index(out)
    assert len(reloaded) == len(records)
    assert reloaded[0].subject_id == records[0].subject_id
    assert reloaded[1].age == records[1].age
