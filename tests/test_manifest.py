"""Tests for the manifest system.

Manifests are what make "reproducible" a property of the repository rather than
a property of intent, so their behaviour is pinned down here.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from satai.errors import ValidationError
from satai.providers.base import SceneRef, SearchQuery, SearchResult
from satai.providers.manifest import Manifest, sha256_file

BIHAR_BBOX = (85.0, 25.2, 86.5, 26.2)


@pytest.fixture
def manifest() -> Manifest:
    return Manifest(manifest_id="test_run", aoi_id="bihar_ganga", purpose="unit test")


@pytest.fixture
def search_result() -> SearchResult:
    return SearchResult(
        provider="cdse",
        query=SearchQuery(
            bbox=BIHAR_BBOX,
            start=date(2024, 6, 1),
            end=date(2024, 10, 31),
            collection="SENTINEL-1",
        ),
        scenes=[
            SceneRef(scene_id="S1A_001", collection="SENTINEL-1", provider="cdse"),
            SceneRef(scene_id="S1A_002", collection="SENTINEL-1", provider="cdse"),
        ],
    )


def test_records_scene_ids(manifest: Manifest, search_result: SearchResult) -> None:
    manifest.record_search(search_result)
    assert manifest.scene_ids == ["S1A_001", "S1A_002"]
    assert len(manifest.queries) == 1


def test_scene_ids_are_deduplicated(manifest: Manifest, search_result: SearchResult) -> None:
    manifest.record_search(search_result)
    manifest.record_search(search_result)
    assert manifest.scene_ids == ["S1A_001", "S1A_002"]


def test_truncated_search_raises_a_warning(manifest: Manifest) -> None:
    """A truncated search changes how the count should be read, so it propagates."""
    result = SearchResult(
        provider="cdse",
        query=SearchQuery(
            bbox=BIHAR_BBOX,
            start=date(2024, 6, 1),
            end=date(2024, 10, 31),
            collection="SENTINEL-1",
            limit=10,
        ),
        scenes=[],
        truncated=True,
    )
    manifest.record_search(result)
    assert any("lower bound" in w for w in manifest.warnings)


def test_add_file_hashes_and_records(manifest: Manifest, tmp_path: Path) -> None:
    data_file = tmp_path / "chip.tif"
    data_file.write_bytes(b"fake raster bytes")

    entry = manifest.add_file(
        data_file, dataset="COPERNICUS/S1_GRD", provider="gee", data_root=tmp_path
    )
    assert entry.path == "chip.tif"
    assert entry.bytes == len(b"fake raster bytes")
    assert entry.sha256 == sha256_file(data_file)
    assert manifest.total_bytes == entry.bytes


def test_add_missing_file_raises(manifest: Manifest, tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="missing file"):
        manifest.add_file(tmp_path / "nope.tif", dataset="x", provider="y", data_root=tmp_path)


class TestContentHash:
    def test_ignores_wall_clock_time(self, search_result: SearchResult) -> None:
        """Two runs over the same scenes with the same code should agree.

        If the hash moved with the clock it would be useless for deciding
        whether a result needs recomputing.
        """
        a = Manifest(manifest_id="a", purpose="p", git_sha="abc123")
        b = Manifest(manifest_id="b", purpose="p", git_sha="abc123")
        a.record_search(search_result)
        b.record_search(search_result)
        assert a.content_hash() == b.content_hash()

    def test_changes_with_inputs(self, search_result: SearchResult) -> None:
        a = Manifest(manifest_id="a", purpose="p", git_sha="abc123")
        b = Manifest(manifest_id="b", purpose="p", git_sha="abc123")
        a.record_search(search_result)
        b.record_search(search_result)
        b.scene_ids.append("S1A_003")
        assert a.content_hash() != b.content_hash()

    def test_changes_with_code_version(self, search_result: SearchResult) -> None:
        """A different commit can produce a different result from identical inputs."""
        a = Manifest(manifest_id="a", purpose="p", git_sha="abc123")
        b = Manifest(manifest_id="b", purpose="p", git_sha="def456")
        a.record_search(search_result)
        b.record_search(search_result)
        assert a.content_hash() != b.content_hash()


class TestRoundTrip:
    def test_write_and_load(
        self, manifest: Manifest, search_result: SearchResult, tmp_path: Path
    ) -> None:
        manifest.record_search(search_result)
        manifest.warn("provider fell back from gee to cdse")
        path = manifest.write(tmp_path)

        loaded = Manifest.load(path)
        assert loaded.manifest_id == manifest.manifest_id
        assert loaded.scene_ids == manifest.scene_ids
        assert loaded.warnings == manifest.warnings
        assert loaded.content_hash() == manifest.content_hash()

    def test_load_missing_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError, match="not found"):
            Manifest.load(tmp_path / "nope.json")


class TestVerify:
    def test_clean_when_files_match(self, manifest: Manifest, tmp_path: Path) -> None:
        f = tmp_path / "a.tif"
        f.write_bytes(b"content")
        manifest.add_file(f, dataset="d", provider="p", data_root=tmp_path)
        assert manifest.verify(tmp_path) == []

    def test_detects_deletion(self, manifest: Manifest, tmp_path: Path) -> None:
        f = tmp_path / "a.tif"
        f.write_bytes(b"content")
        manifest.add_file(f, dataset="d", provider="p", data_root=tmp_path)
        f.unlink()
        assert any("missing" in p for p in manifest.verify(tmp_path))

    def test_detects_corruption_at_same_size(self, manifest: Manifest, tmp_path: Path) -> None:
        """A partially-rewritten raster of identical length must not pass.

        This is the realistic failure: an interrupted download or a flipped byte
        produces a file that looks right and trains a model whose results cannot
        be explained afterwards.
        """
        f = tmp_path / "a.tif"
        f.write_bytes(b"content")
        manifest.add_file(f, dataset="d", provider="p", data_root=tmp_path)
        f.write_bytes(b"corrupt")  # same length, different bytes
        assert any("checksum mismatch" in p for p in manifest.verify(tmp_path))
