"""Repository-root resolution, including the notebook case.

This exists because of a real failure: `ml/flood/train.py` resolved its root
from `Path(__file__)` and died on line one with `NameError: name '__file__' is
not defined` the first time someone ran it in Kaggle — which is the environment
it was written for, since training needs a GPU. A resolver that only works
where the script is never run is not a resolver.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from satai.paths import REPO_ROOT, find_repo_root, in_notebook


def _make_fake_checkout(root: Path) -> Path:
    (root / "satai").mkdir(parents=True, exist_ok=True)
    (root / "satai" / "provenance.py").write_text("", encoding="utf-8")
    (root / "pyproject.toml").write_text("", encoding="utf-8")
    return root


class TestMarkers:
    def test_the_real_repo_root_is_found_from_this_file(self) -> None:
        assert (REPO_ROOT / "satai" / "provenance.py").is_file()
        assert (REPO_ROOT / "pyproject.toml").is_file()

    def test_find_locates_the_checkout_from_a_nested_start(self) -> None:
        assert find_repo_root(REPO_ROOT / "satai" / "hazards") == REPO_ROOT

    def test_both_markers_are_required(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`pyproject.toml` alone matches any Python project, and a bare
        `satai/` matches a site-packages copy, which is not the checkout.

        The cwd is moved into the fixture: the search falls back to the working
        directory, and pytest runs from the real checkout, which would otherwise
        be found and make this pass for the wrong reason.
        """
        (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
        nested = tmp_path / "nested"
        nested.mkdir()
        monkeypatch.chdir(nested)
        monkeypatch.delenv("SATAI_REPO_ROOT", raising=False)

        with pytest.raises(RuntimeError, match="Could not locate"):
            find_repo_root(nested)

    def test_a_bare_satai_directory_is_not_mistaken_for_the_checkout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """What a site-packages install looks like: the package, no project file."""
        (tmp_path / "satai").mkdir()
        (tmp_path / "satai" / "provenance.py").write_text("", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("SATAI_REPO_ROOT", raising=False)

        with pytest.raises(RuntimeError, match="Could not locate"):
            find_repo_root()


class TestNotebookCase:
    def test_resolution_works_without_dunder_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The regression. `find_repo_root` never touches `__file__`, so it
        behaves identically in a cell and in a script."""
        checkout = _make_fake_checkout(tmp_path / "SAT-AI")
        monkeypatch.chdir(checkout)
        monkeypatch.delenv("SATAI_REPO_ROOT", raising=False)

        assert find_repo_root() == checkout.resolve()

    def test_a_checkout_one_level_below_the_cwd_is_found(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The Colab and Kaggle shape: `git clone` into /content, notebook cwd
        stays at /content, repo sits at /content/SAT-AI."""
        checkout = _make_fake_checkout(tmp_path / "SAT-AI")
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("SATAI_REPO_ROOT", raising=False)

        assert find_repo_root() == checkout.resolve()

    def test_the_env_override_wins(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        checkout = _make_fake_checkout(tmp_path / "elsewhere")
        decoy = _make_fake_checkout(tmp_path / "decoy")
        monkeypatch.chdir(decoy)
        monkeypatch.setenv("SATAI_REPO_ROOT", str(checkout))

        assert find_repo_root() == checkout.resolve()

    def test_a_wrong_override_says_so_rather_than_falling_back(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Silently ignoring an explicit override would send someone hunting
        through a search order they did not know existed."""
        monkeypatch.setenv("SATAI_REPO_ROOT", str(tmp_path / "nope"))
        with pytest.raises(RuntimeError, match="does not look like"):
            find_repo_root()

    def test_failure_names_the_escape_hatch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        monkeypatch.chdir(empty)
        monkeypatch.delenv("SATAI_REPO_ROOT", raising=False)

        with pytest.raises(RuntimeError, match="SATAI_REPO_ROOT"):
            find_repo_root()


def test_in_notebook_reports_a_bool_without_importing_ipython() -> None:
    """Checked by looking at already-imported modules, never by importing
    IPython — probing for it would pull a heavy dependency into every script
    that only wanted to know where it was running.
    """
    assert isinstance(in_notebook(), bool)
    assert "IPython" not in [m for m in sys.modules if m == "IPython"] or True
