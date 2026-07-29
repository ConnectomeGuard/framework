"""
Tests for raw_site_archive utility — Phase 2R archiving safety.

Covers:
  - DTI sites are never selected for archiving
  - dry-run makes no filesystem changes
  - compression creates archive + SHA-256 checksum file
  - delete requires --confirm-delete=True
  - delete blocked if archive missing or checksum missing
  - checksum mismatch blocks delete
  - archive report CSV and JSON are generated with required columns
  - report is written to the archive directory (not raw_root)
"""

from __future__ import annotations

import csv
import json
import shutil
import tarfile
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neurofiber.utils.raw_site_archive import (
    DTI_SITE_FOLDERS,
    NON_DTI_SITE_FOLDERS,
    ArchiveRecord,
    SiteInfo,
    assert_not_dti,
    compress_site,
    delete_source_after_compress,
    display_to_folder,
    dry_run_site,
    folder_to_display,
    measure_site,
    move_site,
    scan_non_dti_sites,
    sha256_file,
    write_report,
)


# ──────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────

def _make_site_dir(parent: Path, name: str, n_files: int = 3) -> Path:
    """Create a fake site directory with some dummy files."""
    site_dir = parent / name
    site_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_files):
        subj = site_dir / f"ABIDE2-{name.upper()}_1" / f"9{i:04d}" / "session_1" / "dti_1"
        subj.mkdir(parents=True, exist_ok=True)
        (subj / "dti.nii.gz").write_bytes(b"fake nifti data " * 100)
        (subj / "dti.bval").write_text("0 1000\n")
        (subj / "dti.bvec").write_text("1 0\n0 1\n0 0\n")
    return site_dir


def _make_info(site_dir: Path, display: str, folder: str) -> SiteInfo:
    return measure_site(site_dir, display, folder)


# ──────────────────────────────────────────────────────────
# Site registry sanity
# ──────────────────────────────────────────────────────────

class TestSiteRegistry:
    def test_dti_folders_never_in_non_dti(self):
        assert DTI_SITE_FOLDERS.isdisjoint(NON_DTI_SITE_FOLDERS), (
            "A folder appears in both DTI and non-DTI sets!"
        )

    def test_expected_dti_folders_present(self):
        for f in ("bni", "ip", "nyu1", "nyu2", "sdsu", "tcd"):
            assert f in DTI_SITE_FOLDERS

    def test_expected_non_dti_folders_present(self):
        for f in ("eth", "emc", "gu", "iu", "kki", "kul",
                  "ohsu", "onrc", "ucd", "ucla1", "ucla_long",
                  "upsm_long", "usm"):
            assert f in NON_DTI_SITE_FOLDERS

    def test_display_to_folder_roundtrip(self):
        for display, folder in [
            ("ETH", "eth"), ("EMC", "emc"), ("GU", "gu"),
            ("IU_1", "iu"), ("KKI_1", "kki"), ("KUL_3", "kul"),
            ("OHSU_1", "ohsu"), ("ONRC_2", "onrc"), ("UCD_1", "ucd"),
            ("UCLA_1", "ucla1"), ("UCLA_Long", "ucla_long"),
            ("UPSM_Long", "upsm_long"), ("USM_1", "usm"),
        ]:
            assert display_to_folder(display) == folder


# ──────────────────────────────────────────────────────────
# DTI protection
# ──────────────────────────────────────────────────────────

class TestDTIProtection:
    @pytest.mark.parametrize("folder", sorted(DTI_SITE_FOLDERS))
    def test_assert_not_dti_raises_for_dti_sites(self, folder):
        with pytest.raises(ValueError, match="SAFETY VIOLATION"):
            assert_not_dti(folder)

    @pytest.mark.parametrize("folder", sorted(NON_DTI_SITE_FOLDERS))
    def test_assert_not_dti_passes_for_non_dti(self, folder):
        assert_not_dti(folder)  # must not raise

    def test_dry_run_raises_for_dti_site(self, tmp_path):
        info = SiteInfo(
            display_name="BNI", folder_name="bni",
            source_path=tmp_path / "bni", exists=True,
            size_bytes=1000, file_count=3,
        )
        with pytest.raises(ValueError, match="SAFETY VIOLATION"):
            dry_run_site(info)

    def test_move_raises_for_dti_site(self, tmp_path):
        info = SiteInfo(
            display_name="NYU_1", folder_name="nyu1",
            source_path=tmp_path / "nyu1", exists=True,
            size_bytes=0, file_count=0,
        )
        with pytest.raises(ValueError, match="SAFETY VIOLATION"):
            move_site(info, tmp_path / "archive")

    def test_compress_raises_for_dti_site(self, tmp_path):
        info = SiteInfo(
            display_name="TCD_1", folder_name="tcd",
            source_path=tmp_path / "tcd", exists=True,
            size_bytes=0, file_count=0,
        )
        with pytest.raises(ValueError, match="SAFETY VIOLATION"):
            compress_site(info, tmp_path / "archive")

    def test_scan_non_dti_never_includes_dti(self, tmp_path):
        # Create fake raw_root with both DTI and non-DTI dirs
        for f in list(DTI_SITE_FOLDERS) + list(NON_DTI_SITE_FOLDERS):
            (tmp_path / f).mkdir(parents=True, exist_ok=True)
        sites = scan_non_dti_sites(tmp_path)
        folders_found = {s.folder_name for s in sites}
        assert folders_found.isdisjoint(DTI_SITE_FOLDERS), (
            f"DTI sites appeared in scan: {folders_found & DTI_SITE_FOLDERS}"
        )


# ──────────────────────────────────────────────────────────
# Dry run — no filesystem changes
# ──────────────────────────────────────────────────────────

class TestDryRun:
    def test_dry_run_returns_record(self, tmp_path):
        site_dir = _make_site_dir(tmp_path, "eth")
        info = _make_info(site_dir, "ETH", "eth")
        rec  = dry_run_site(info)
        assert rec.action == "dry_run"
        assert rec.status == "dry_run"

    def test_dry_run_does_not_move_anything(self, tmp_path):
        site_dir = _make_site_dir(tmp_path, "emc")
        info = _make_info(site_dir, "EMC", "emc")

        files_before = set(tmp_path.rglob("*"))
        dry_run_site(info)
        files_after = set(tmp_path.rglob("*"))

        assert files_before == files_after

    def test_dry_run_reports_size(self, tmp_path):
        site_dir = _make_site_dir(tmp_path, "gu")
        info = _make_info(site_dir, "GU", "gu")
        assert info.size_bytes > 0
        rec = dry_run_site(info)
        assert rec.original_size_gb >= 0

    def test_dry_run_nonexistent_site(self, tmp_path):
        info = SiteInfo(
            display_name="ETH", folder_name="eth",
            source_path=tmp_path / "eth", exists=False,
            size_bytes=0, file_count=0,
        )
        rec = dry_run_site(info)
        assert rec.action == "dry_run"
        assert rec.original_size_gb == 0.0


# ──────────────────────────────────────────────────────────
# Move
# ──────────────────────────────────────────────────────────

class TestMove:
    def test_move_relocates_directory(self, tmp_path):
        site_dir  = _make_site_dir(tmp_path / "raw", "usm")
        info      = _make_info(site_dir, "USM_1", "usm")
        archive   = tmp_path / "archive"

        rec = move_site(info, archive)

        assert rec.status == "success"
        assert not site_dir.exists()
        assert (archive / "usm").exists()

    def test_move_skips_if_dest_exists(self, tmp_path):
        site_dir = _make_site_dir(tmp_path / "raw", "ucd")
        info     = _make_info(site_dir, "UCD_1", "ucd")
        archive  = tmp_path / "archive"
        (archive / "ucd").mkdir(parents=True)  # pre-create dest

        rec = move_site(info, archive)
        assert rec.status == "skipped"
        assert site_dir.exists()  # source not touched

    def test_move_skips_missing_source(self, tmp_path):
        info = SiteInfo(
            display_name="IU_1", folder_name="iu",
            source_path=tmp_path / "iu", exists=False,
            size_bytes=0, file_count=0,
        )
        rec = move_site(info, tmp_path / "archive")
        assert rec.status == "skipped"

    def test_move_does_not_touch_dti_sites(self, tmp_path):
        info = SiteInfo(
            display_name="BNI", folder_name="bni",
            source_path=tmp_path / "bni", exists=True,
            size_bytes=0, file_count=0,
        )
        with pytest.raises(ValueError):
            move_site(info, tmp_path / "archive")


# ──────────────────────────────────────────────────────────
# Compress
# ──────────────────────────────────────────────────────────

class TestCompress:
    def test_compress_creates_tar_gz(self, tmp_path):
        site_dir = _make_site_dir(tmp_path / "raw", "kki")
        info     = _make_info(site_dir, "KKI_1", "kki")
        archive  = tmp_path / "archive"

        rec = compress_site(info, archive)

        assert rec.status == "success"
        tar_path = archive / "kki.tar.gz"
        assert tar_path.exists()
        assert tar_path.stat().st_size > 0  # file has bytes; too small to register as GB

    def test_compress_creates_sha256_checksum(self, tmp_path):
        site_dir = _make_site_dir(tmp_path / "raw", "ohsu")
        info     = _make_info(site_dir, "OHSU_1", "ohsu")
        archive  = tmp_path / "archive"

        rec = compress_site(info, archive)

        checksum_path = archive / "ohsu.tar.gz.sha256"
        assert rec.status == "success"
        assert checksum_path.exists()
        line = checksum_path.read_text().strip()
        assert len(line.split()[0]) == 64   # SHA-256 hex digest length

    def test_compress_checksum_matches_archive(self, tmp_path):
        site_dir = _make_site_dir(tmp_path / "raw", "onrc")
        info     = _make_info(site_dir, "ONRC_2", "onrc")
        archive  = tmp_path / "archive"

        rec = compress_site(info, archive)

        expected_digest = rec.sha256
        actual_digest   = sha256_file(archive / "onrc.tar.gz")
        assert expected_digest == actual_digest

    def test_compress_archive_is_valid_tar(self, tmp_path):
        site_dir = _make_site_dir(tmp_path / "raw", "ucla1")
        info     = _make_info(site_dir, "UCLA_1", "ucla1")
        archive  = tmp_path / "archive"

        compress_site(info, archive)

        assert tarfile.is_tarfile(str(archive / "ucla1.tar.gz"))

    def test_compress_source_still_exists(self, tmp_path):
        """Compression alone never deletes the source."""
        site_dir = _make_site_dir(tmp_path / "raw", "upsm_long")
        info     = _make_info(site_dir, "UPSM_Long", "upsm_long")
        archive  = tmp_path / "archive"

        compress_site(info, archive)
        assert site_dir.exists()

    def test_compress_skips_if_archive_exists(self, tmp_path):
        site_dir = _make_site_dir(tmp_path / "raw", "ucla_long")
        info     = _make_info(site_dir, "UCLA_Long", "ucla_long")
        archive  = tmp_path / "archive"
        archive.mkdir(parents=True)
        (archive / "ucla_long.tar.gz").write_bytes(b"existing")  # pre-create

        rec = compress_site(info, archive)
        assert rec.status == "skipped"
        assert site_dir.exists()

    def test_compress_missing_source(self, tmp_path):
        info = SiteInfo(
            display_name="KUL_3", folder_name="kul",
            source_path=tmp_path / "kul", exists=False,
            size_bytes=0, file_count=0,
        )
        rec = compress_site(info, tmp_path / "archive")
        assert rec.status == "skipped"


# ──────────────────────────────────────────────────────────
# Delete after compress
# ──────────────────────────────────────────────────────────

class TestDeleteAfterCompress:
    def test_delete_requires_confirm_delete_flag(self, tmp_path):
        site_dir = _make_site_dir(tmp_path / "raw", "eth")
        info     = _make_info(site_dir, "ETH", "eth")
        archive  = tmp_path / "archive"
        compress_site(info, archive)

        rec = delete_source_after_compress(info, archive, confirm_delete=False)

        assert rec.status == "skipped"
        assert site_dir.exists()  # source NOT deleted
        assert "--confirm-delete" in (rec.error_message or "")

    def test_delete_succeeds_with_confirm(self, tmp_path):
        site_dir = _make_site_dir(tmp_path / "raw", "emc")
        info     = _make_info(site_dir, "EMC", "emc")
        archive  = tmp_path / "archive"
        compress_site(info, archive)

        rec = delete_source_after_compress(info, archive, confirm_delete=True)

        assert rec.status == "success"
        assert not site_dir.exists()

    def test_delete_blocked_if_archive_missing(self, tmp_path):
        site_dir = _make_site_dir(tmp_path / "raw", "gu")
        info     = _make_info(site_dir, "GU", "gu")
        archive  = tmp_path / "archive"
        # Do NOT compress — no archive exists

        rec = delete_source_after_compress(info, archive, confirm_delete=True)

        assert rec.status == "failed"
        assert site_dir.exists()  # source intact

    def test_delete_blocked_if_checksum_missing(self, tmp_path):
        site_dir = _make_site_dir(tmp_path / "raw", "iu")
        info     = _make_info(site_dir, "IU_1", "iu")
        archive  = tmp_path / "archive"
        archive.mkdir(parents=True)
        # Write archive but not checksum
        (archive / "iu.tar.gz").write_bytes(b"fake archive")

        rec = delete_source_after_compress(info, archive, confirm_delete=True)

        assert rec.status == "failed"
        assert "Checksum" in rec.error_message
        assert site_dir.exists()

    def test_delete_blocked_if_checksum_mismatch(self, tmp_path):
        site_dir = _make_site_dir(tmp_path / "raw", "usm")
        info     = _make_info(site_dir, "USM_1", "usm")
        archive  = tmp_path / "archive"
        compress_site(info, archive)

        # Corrupt the archive after compression
        archive_path = archive / "usm.tar.gz"
        archive_path.write_bytes(b"tampered content")

        rec = delete_source_after_compress(info, archive, confirm_delete=True)

        assert rec.status == "failed"
        assert "mismatch" in rec.error_message
        assert site_dir.exists()

    def test_delete_blocked_for_dti_site(self, tmp_path):
        info = SiteInfo(
            display_name="BNI", folder_name="bni",
            source_path=tmp_path / "bni", exists=True,
            size_bytes=0, file_count=0,
        )
        with pytest.raises(ValueError, match="SAFETY VIOLATION"):
            delete_source_after_compress(info, tmp_path / "archive", confirm_delete=True)


# ──────────────────────────────────────────────────────────
# Archive report
# ──────────────────────────────────────────────────────────

class TestArchiveReport:
    REQUIRED_COLS = [
        "site", "source_path", "archive_path", "action",
        "original_size_gb", "archive_size_gb", "sha256",
        "status", "error_message", "timestamp",
    ]

    def _make_records(self) -> list[ArchiveRecord]:
        return [
            ArchiveRecord(
                site="ETH", source_path="/raw/eth",
                archive_path="/archive/eth.tar.gz",
                action="compressed", original_size_gb=1.6,
                archive_size_gb=0.8, sha256="abc123", status="success",
            ),
            ArchiveRecord(
                site="GU", source_path="/raw/gu",
                archive_path="/archive/gu.tar.gz",
                action="compressed", original_size_gb=4.2,
                archive_size_gb=0.0, sha256="",
                status="failed", error_message="Disk full",
            ),
        ]

    def test_report_csv_created(self, tmp_path):
        records  = self._make_records()
        csv_path, _ = write_report(records, tmp_path)
        assert csv_path.exists()

    def test_report_json_created(self, tmp_path):
        records = self._make_records()
        _, json_path = write_report(records, tmp_path)
        assert json_path.exists()

    def test_report_csv_has_required_columns(self, tmp_path):
        records = self._make_records()
        csv_path, _ = write_report(records, tmp_path)
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames
        for col in self.REQUIRED_COLS:
            assert col in cols, f"Missing column in CSV: {col}"

    def test_report_json_valid_and_has_required_keys(self, tmp_path):
        records = self._make_records()
        _, json_path = write_report(records, tmp_path)
        data = json.loads(json_path.read_text())
        assert isinstance(data, list)
        assert len(data) == 2
        for row in data:
            for key in self.REQUIRED_COLS:
                assert key in row, f"Missing key in JSON: {key}"

    def test_report_row_count_matches_records(self, tmp_path):
        records = self._make_records()
        csv_path, _ = write_report(records, tmp_path)
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == len(records)

    def test_report_written_to_archive_not_raw(self, tmp_path):
        raw_root = tmp_path / "raw"
        archive_root = tmp_path / "archive"
        records = self._make_records()
        csv_path, json_path = write_report(records, archive_root)
        # Report must be in archive dir, not raw dir
        assert str(raw_root) not in str(csv_path)
        assert str(archive_root) in str(csv_path)


# ──────────────────────────────────────────────────────────
# SHA-256 helper
# ──────────────────────────────────────────────────────────

class TestSha256:
    def test_sha256_deterministic(self, tmp_path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"hello world")
        assert sha256_file(f) == sha256_file(f)

    def test_sha256_changes_with_content(self, tmp_path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"hello world")
        d1 = sha256_file(f)
        f.write_bytes(b"goodbye world")
        d2 = sha256_file(f)
        assert d1 != d2

    def test_sha256_known_value(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_bytes(b"")
        # SHA-256 of empty string
        assert sha256_file(f) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


# ──────────────────────────────────────────────────────────
# Measure site
# ──────────────────────────────────────────────────────────

class TestMeasureSite:
    def test_measure_existing_site(self, tmp_path):
        site_dir = _make_site_dir(tmp_path, "kki")
        info = measure_site(site_dir, "KKI_1", "kki")
        assert info.exists is True
        assert info.size_bytes > 0
        assert info.file_count > 0

    def test_measure_nonexistent_site(self, tmp_path):
        info = measure_site(tmp_path / "nonexistent", "ETH", "eth")
        assert info.exists is False
        assert info.size_bytes == 0
        assert info.file_count == 0

    def test_is_dti_flag_correct(self, tmp_path):
        dti_info = SiteInfo(
            display_name="BNI", folder_name="bni",
            source_path=tmp_path, exists=True, size_bytes=0, file_count=0,
        )
        non_dti_info = SiteInfo(
            display_name="ETH", folder_name="eth",
            source_path=tmp_path, exists=True, size_bytes=0, file_count=0,
        )
        assert dti_info.is_dti is True
        assert non_dti_info.is_dti is False
