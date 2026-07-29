"""
Raw site archiving utilities for NeuroFiber ABIDE-II data.

Safety contract:
  - DTI_SITE_FOLDERS is the authoritative whitelist. These folders are NEVER touched.
  - dry-run is the default; no filesystem mutations happen unless explicitly requested.
  - delete is only possible after successful compression + checksum + --confirm-delete flag.
  - data/raw/abide_ii/ itself is never removed, only subdirectories within it.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import tarfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ──────────────────────────────────────────────────────────
# Site registry
# ──────────────────────────────────────────────────────────

# DTI-capable sites — NEVER touch these
DTI_SITE_FOLDERS: frozenset[str] = frozenset({
    "bni", "ip", "nyu1", "nyu2", "sdsu", "tcd",
})

# Display-name → folder mapping for non-DTI sites
_NON_DTI_DISPLAY_TO_FOLDER: dict[str, str] = {
    "ETH":        "eth",
    "EMC":        "emc",
    "GU":         "gu",
    "IU_1":       "iu",
    "KKI_1":      "kki",
    "KUL_3":      "kul",
    "OHSU_1":     "ohsu",
    "ONRC_2":     "onrc",
    "UCD_1":      "ucd",
    "UCLA_1":     "ucla1",
    "UCLA_Long":  "ucla_long",
    "UPSM_Long":  "upsm_long",
    "USM_1":      "usm",
}

NON_DTI_SITE_FOLDERS: frozenset[str] = frozenset(_NON_DTI_DISPLAY_TO_FOLDER.values())


def display_to_folder(display_name: str) -> str:
    return _NON_DTI_DISPLAY_TO_FOLDER.get(display_name, display_name.lower())


def folder_to_display(folder: str) -> str:
    inv = {v: k for k, v in _NON_DTI_DISPLAY_TO_FOLDER.items()}
    return inv.get(folder, folder.upper())


# ──────────────────────────────────────────────────────────
# Data models
# ──────────────────────────────────────────────────────────

@dataclass
class SiteInfo:
    display_name:   str
    folder_name:    str
    source_path:    Path
    exists:         bool
    size_bytes:     int     = 0
    file_count:     int     = 0

    @property
    def size_gb(self) -> float:
        return round(self.size_bytes / 1024 ** 3, 3)

    @property
    def is_dti(self) -> bool:
        return self.folder_name in DTI_SITE_FOLDERS


@dataclass
class ArchiveRecord:
    site:               str
    source_path:        str
    archive_path:       str
    action:             str           # dry_run | moved | compressed | deleted | skipped | failed
    original_size_gb:   float
    archive_size_gb:    float         = 0.0
    sha256:             str           = ""
    status:             str           = "pending"
    error_message:      Optional[str] = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return asdict(self)


# ──────────────────────────────────────────────────────────
# Size measurement
# ──────────────────────────────────────────────────────────

def measure_site(source_path: Path, display_name: str, folder_name: str) -> SiteInfo:
    exists = source_path.exists() and source_path.is_dir()
    size_bytes = 0
    file_count = 0
    if exists:
        for p in source_path.rglob("*"):
            if p.is_file():
                size_bytes += p.stat().st_size
                file_count += 1
    return SiteInfo(
        display_name=display_name,
        folder_name=folder_name,
        source_path=source_path,
        exists=exists,
        size_bytes=size_bytes,
        file_count=file_count,
    )


def scan_non_dti_sites(raw_root: Path) -> list[SiteInfo]:
    """
    Measure all non-DTI site folders under raw_root.
    DTI sites are not included in the returned list.
    """
    sites = []
    for display, folder in sorted(_NON_DTI_DISPLAY_TO_FOLDER.items()):
        src = raw_root / folder
        info = measure_site(src, display, folder)
        sites.append(info)
    return sites


# ──────────────────────────────────────────────────────────
# Safety guards
# ──────────────────────────────────────────────────────────

def assert_not_dti(folder_name: str) -> None:
    if folder_name in DTI_SITE_FOLDERS:
        raise ValueError(
            f"SAFETY VIOLATION: '{folder_name}' is a DTI site and must never be archived."
        )


def assert_not_raw_root(path: Path, raw_root: Path) -> None:
    if path.resolve() == raw_root.resolve():
        raise ValueError(
            f"SAFETY VIOLATION: refusing to operate on raw_root itself ({raw_root}). "
            "Only site subdirectories may be archived."
        )


# ──────────────────────────────────────────────────────────
# Checksum
# ──────────────────────────────────────────────────────────

def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute SHA-256 of a file. Streams in chunks to handle large archives."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


# ──────────────────────────────────────────────────────────
# Actions
# ──────────────────────────────────────────────────────────

def dry_run_site(info: SiteInfo) -> ArchiveRecord:
    """Return a report record without touching the filesystem."""
    assert_not_dti(info.folder_name)
    return ArchiveRecord(
        site=info.display_name,
        source_path=str(info.source_path),
        archive_path="(not created — dry run)",
        action="dry_run",
        original_size_gb=info.size_gb,
        status="dry_run",
    )


def move_site(
    info:        SiteInfo,
    archive_root: Path,
) -> ArchiveRecord:
    """
    Move site directory from raw_root to archive_root.
    archive_root: e.g. data/archive/raw_non_dti/abide_ii/
    """
    assert_not_dti(info.folder_name)
    dest = archive_root / info.folder_name
    rec = ArchiveRecord(
        site=info.display_name,
        source_path=str(info.source_path),
        archive_path=str(dest),
        action="moved",
        original_size_gb=info.size_gb,
    )

    if not info.exists:
        rec.status = "skipped"
        rec.error_message = "Source directory does not exist"
        return rec

    if dest.exists():
        rec.status = "skipped"
        rec.error_message = f"Destination already exists: {dest}"
        return rec

    try:
        archive_root.mkdir(parents=True, exist_ok=True)
        shutil.move(str(info.source_path), str(dest))
        rec.archive_size_gb = info.size_gb  # same data, just moved
        rec.status = "success"
    except Exception as exc:
        rec.status = "failed"
        rec.error_message = str(exc)

    return rec


def compress_site(
    info:         SiteInfo,
    archive_root: Path,
) -> ArchiveRecord:
    """
    Compress site directory to a .tar.gz archive and write a SHA-256 checksum file.
    Does NOT delete the source.
    archive_root: e.g. data/archive/raw_non_dti/abide_ii/
    """
    assert_not_dti(info.folder_name)
    archive_path = archive_root / f"{info.folder_name}.tar.gz"
    checksum_path = archive_root / f"{info.folder_name}.tar.gz.sha256"

    rec = ArchiveRecord(
        site=info.display_name,
        source_path=str(info.source_path),
        archive_path=str(archive_path),
        action="compressed",
        original_size_gb=info.size_gb,
    )

    if not info.exists:
        rec.status = "skipped"
        rec.error_message = "Source directory does not exist"
        return rec

    if archive_path.exists():
        rec.status = "skipped"
        rec.error_message = f"Archive already exists: {archive_path}"
        return rec

    try:
        archive_root.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(str(info.source_path), arcname=info.folder_name)
        elapsed = time.time() - t0

        digest = sha256_file(archive_path)
        checksum_path.write_text(f"{digest}  {archive_path.name}\n")

        rec.archive_size_gb = round(archive_path.stat().st_size / 1024 ** 3, 3)
        rec.sha256   = digest
        rec.status   = "success"
    except Exception as exc:
        # Clean up partial archive if compression failed midway
        if archive_path.exists():
            archive_path.unlink(missing_ok=True)
        rec.status = "failed"
        rec.error_message = str(exc)

    return rec


def delete_source_after_compress(
    info:            SiteInfo,
    archive_root:    Path,
    confirm_delete:  bool,
) -> ArchiveRecord:
    """
    Delete the source directory ONLY IF:
      1. The .tar.gz exists in archive_root
      2. The .sha256 checksum exists
      3. confirm_delete=True (--confirm-delete flag passed by user)
    """
    assert_not_dti(info.folder_name)
    archive_path  = archive_root / f"{info.folder_name}.tar.gz"
    checksum_path = archive_root / f"{info.folder_name}.tar.gz.sha256"

    rec = ArchiveRecord(
        site=info.display_name,
        source_path=str(info.source_path),
        archive_path=str(archive_path),
        action="deleted",
        original_size_gb=info.size_gb,
    )

    if not confirm_delete:
        rec.status = "skipped"
        rec.error_message = "--confirm-delete not passed; source NOT deleted"
        return rec

    if not info.exists:
        rec.status = "skipped"
        rec.error_message = "Source directory does not exist (already deleted?)"
        return rec

    if not archive_path.exists():
        rec.status = "failed"
        rec.error_message = f"Archive not found: {archive_path} — refusing to delete source"
        return rec

    if not checksum_path.exists():
        rec.status = "failed"
        rec.error_message = f"Checksum file missing: {checksum_path} — refusing to delete source"
        return rec

    # Verify checksum before deleting
    expected_line = checksum_path.read_text().strip()
    expected_digest = expected_line.split()[0]
    actual_digest = sha256_file(archive_path)
    if actual_digest != expected_digest:
        rec.status = "failed"
        rec.error_message = (
            f"Checksum mismatch for {archive_path.name}: "
            f"expected {expected_digest[:16]}… got {actual_digest[:16]}… — "
            "refusing to delete source"
        )
        return rec

    try:
        shutil.rmtree(str(info.source_path))
        rec.sha256 = actual_digest
        rec.status = "success"
    except Exception as exc:
        rec.status = "failed"
        rec.error_message = str(exc)

    return rec


# ──────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────

def write_report(
    records:      list[ArchiveRecord],
    archive_root: Path,
) -> tuple[Path, Path]:
    """
    Write archive_report.csv and archive_report.json to archive_root.
    Returns (csv_path, json_path).
    """
    archive_root.mkdir(parents=True, exist_ok=True)
    csv_path  = archive_root / "archive_report.csv"
    json_path = archive_root / "archive_report.json"

    rows = [r.to_dict() for r in records]

    fieldnames = [
        "site", "source_path", "archive_path", "action",
        "original_size_gb", "archive_size_gb", "sha256",
        "status", "error_message", "timestamp",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    json_path.write_text(json.dumps(rows, indent=2))

    return csv_path, json_path
