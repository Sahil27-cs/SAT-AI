"""Manifests: what was downloaded, from where, and exactly when.

A manifest is written beside every derived artifact and records the scenes,
queries, code version and checksums that produced it. It is the mechanism that
turns "we used Sentinel-1 over Bihar" into a statement someone else can act on.

Manifests are committed to git; the data they describe is not. A cloned
repository therefore carries a complete record of what it would take to
reproduce every result, at a few kilobytes instead of a few hundred gigabytes.

This is also where the evaluation chapter's reproducibility claim comes from.
"Reproducible" is not a property of intent; it is a property of having written
down the scene identifiers.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from satai.errors import ValidationError
from satai.logging import get_logger
from satai.providers.base import SearchResult

log = get_logger(__name__)

__all__ = ["Manifest", "ManifestEntry", "current_git_sha", "sha256_file"]

_HASH_CHUNK = 1 << 20  # 1 MiB


def sha256_file(path: Path) -> str:
    """SHA-256 of a file, streamed so a 2 GB raster does not enter memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def current_git_sha(repo_root: Path | None = None) -> str | None:
    """Short commit SHA, or ``None`` outside a git checkout.

    Recorded in every manifest so an artifact can be traced to the code that
    made it. A result whose code version is unknown is not reproducible,
    whatever else was recorded.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            cwd=repo_root,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() or None if result.returncode == 0 else None


class ManifestEntry(BaseModel):
    """One file, and the observation it came from."""

    model_config = ConfigDict(frozen=True)

    path: str = Field(description="Path relative to the data directory.")
    sha256: str
    bytes: int = Field(ge=0)
    dataset: str
    provider: str
    scene_id: str | None = None
    acquired_at: datetime | None = None
    written_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    notes: str | None = None


class Manifest(BaseModel):
    """The complete record behind one acquisition or processing run."""

    model_config = ConfigDict(frozen=False)  # entries accumulate during a run

    manifest_id: str = Field(description="Stable identifier, e.g. 'bihar_ganga_s1_2024'.")
    aoi_id: str | None = None
    purpose: str = Field(description="What this run was for, in one line.")

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    satai_version: str = Field(default="0.1.0")
    git_sha: str | None = Field(default_factory=current_git_sha)

    queries: list[dict[str, Any]] = Field(default_factory=list)
    entries: list[ManifestEntry] = Field(default_factory=list)
    scene_ids: list[str] = Field(default_factory=list)

    warnings: list[str] = Field(
        default_factory=list,
        description=(
            "Anything that would change how a result should be read: a provider "
            "fallback, a truncated search, a gap in coverage. These propagate "
            "into envelope caveats rather than being silently dropped."
        ),
    )
    stats: dict[str, Any] = Field(default_factory=dict)

    # -- building ----------------------------------------------------------

    def record_search(self, result: SearchResult) -> None:
        """Record a catalogue search and the scenes it returned."""
        self.queries.append(
            {
                "provider": result.provider,
                "query": result.query.model_dump(mode="json"),
                "retrieved_at": result.retrieved_at.isoformat(),
                "n_scenes": len(result.scenes),
                "truncated": result.truncated,
            }
        )
        for scene in result.scenes:
            if scene.scene_id not in self.scene_ids:
                self.scene_ids.append(scene.scene_id)

        if result.truncated:
            self.warnings.append(
                f"{result.provider} search for {result.query.collection} hit the "
                f"result limit ({result.query.limit}); scene counts are a lower bound"
            )

    def add_file(
        self,
        path: Path,
        *,
        dataset: str,
        provider: str,
        data_root: Path,
        scene_id: str | None = None,
        acquired_at: datetime | None = None,
        notes: str | None = None,
    ) -> ManifestEntry:
        """Hash a file and record it. The file must exist."""
        if not path.is_file():
            raise ValidationError(f"cannot add missing file to manifest: {path}")
        try:
            relative = path.relative_to(data_root).as_posix()
        except ValueError:
            relative = path.as_posix()

        entry = ManifestEntry(
            path=relative,
            sha256=sha256_file(path),
            bytes=path.stat().st_size,
            dataset=dataset,
            provider=provider,
            scene_id=scene_id,
            acquired_at=acquired_at,
            notes=notes,
        )
        self.entries.append(entry)
        return entry

    def warn(self, message: str) -> None:
        """Record something that changes how the result should be read."""
        # NB: the key must not be "msg" -- that is reserved on LogRecord and
        # raises KeyError at log time rather than at write time.
        log.warning("manifest warning", extra={"manifest": self.manifest_id, "detail": message})
        self.warnings.append(message)

    # -- identity ----------------------------------------------------------

    def content_hash(self) -> str:
        """Hash over inputs and code version, ignoring wall-clock timestamps.

        Two runs with the same hash used the same scenes and the same code, so
        they should produce the same output. Different hash, different result --
        and the manifest says which input changed.
        """
        payload = {
            "queries": [{k: v for k, v in q.items() if k != "retrieved_at"} for q in self.queries],
            "scene_ids": sorted(self.scene_ids),
            "file_hashes": sorted(e.sha256 for e in self.entries),
            "git_sha": self.git_sha,
            "satai_version": self.satai_version,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]

    @property
    def total_bytes(self) -> int:
        return sum(e.bytes for e in self.entries)

    # -- persistence -------------------------------------------------------

    def write(self, manifests_dir: Path) -> Path:
        """Write to ``<manifests_dir>/<manifest_id>.json``."""
        manifests_dir.mkdir(parents=True, exist_ok=True)
        path = manifests_dir / f"{self.manifest_id}.json"
        payload = self.model_dump(mode="json")
        payload["content_hash"] = self.content_hash()
        path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
        log.info(
            "manifest written",
            extra={
                "path": str(path),
                "scenes": len(self.scene_ids),
                "files": len(self.entries),
                "content_hash": payload["content_hash"],
            },
        )
        return path

    @classmethod
    def load(cls, path: Path) -> Manifest:
        """Read a manifest back."""
        if not path.is_file():
            raise ValidationError(f"manifest not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.pop("content_hash", None)
        return cls.model_validate(payload)

    def verify(self, data_root: Path) -> list[str]:
        """Check every recorded file still exists with its recorded hash.

        Returns a list of problems; empty means the data on disk matches the
        record. Run this before a training run -- a silently corrupted or
        partially-downloaded raster produces a model whose results cannot be
        explained later.
        """
        problems: list[str] = []
        for entry in self.entries:
            path = data_root / entry.path
            if not path.is_file():
                problems.append(f"missing: {entry.path}")
                continue
            if path.stat().st_size != entry.bytes:
                problems.append(
                    f"size changed: {entry.path} "
                    f"({entry.bytes} recorded, {path.stat().st_size} on disk)"
                )
                continue
            if sha256_file(path) != entry.sha256:
                problems.append(f"checksum mismatch: {entry.path}")
        return problems
