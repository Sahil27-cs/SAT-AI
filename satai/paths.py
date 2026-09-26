"""Locating the repository root, from a file or from a notebook.

`Path(__file__)` is the obvious way to find the checkout and it breaks in the
one environment this project most needs it to work in. Training runs on a GPU
notebook (ADR-001), and Jupyter, Colab and Kaggle all execute cells with no
`__file__` defined — so a script that resolves its root that way dies on line
one with a `NameError` that says nothing about what to do.

This module is the canonical resolver. Scripts still carry a small inline
fallback for the case where the package is not installed at all, because they
have to find the root *before* they can import from it; once
`pip install -e ".[dev]"` has run, they import this instead and no searching
happens.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

__all__ = ["REPO_ROOT", "find_repo_root", "in_notebook"]

#: Files that together identify this checkout rather than a same-named directory
#: somewhere else. Both must be present: `pyproject.toml` alone matches any
#: Python project, and a bare `satai/` directory matches a virtualenv's
#: site-packages copy, which is emphatically not the repository.
_MARKERS: tuple[tuple[str, ...], ...] = (
    ("satai", "provenance.py"),
    ("pyproject.toml",),
)


def in_notebook() -> bool:
    """Whether this is running under an IPython kernel."""
    import sys

    return "ipykernel" in sys.modules or "google.colab" in sys.modules


def _is_repo_root(candidate: Path) -> bool:
    return all((candidate.joinpath(*marker)).is_file() for marker in _MARKERS)


def find_repo_root(start: Path | None = None) -> Path:
    """Find the checkout root, or raise with instructions.

    Order of preference, and why each is there:

    1. ``SATAI_REPO_ROOT`` — always wins. A notebook in an unusual layout needs
       an escape hatch that does not require editing the script.
    2. ``start``, when given, and its parents.
    3. The working directory and its parents — a notebook opened inside the
       checkout.
    4. Immediate subdirectories of the working directory — the Colab and Kaggle
       case, where you ``git clone`` into ``/content`` or ``/kaggle/working``
       and the notebook's own directory stays one level above the repo.
    """
    candidates: list[Path] = []

    override = os.environ.get("SATAI_REPO_ROOT")
    if override:
        resolved = Path(override).expanduser().resolve()
        if not _is_repo_root(resolved):
            raise RuntimeError(
                f"SATAI_REPO_ROOT is set to {resolved}, which does not look like "
                f"the SAT-AI checkout: expected satai/provenance.py and "
                f"pyproject.toml inside it."
            )
        return resolved

    if start is not None:
        base = start.resolve()
        candidates.extend([base, *base.parents])

    cwd = Path.cwd().resolve()
    candidates.extend([cwd, *cwd.parents])
    # An unreadable working directory is not worth failing over: the parents
    # above may still hold the answer, and if they do not, the error at the end
    # of this function is the more useful one.
    with contextlib.suppress(OSError):
        candidates.extend(sorted(p for p in cwd.iterdir() if p.is_dir()))

    for candidate in candidates:
        if _is_repo_root(candidate):
            return candidate

    raise RuntimeError(
        "Could not locate the SAT-AI repository root. Run from inside the "
        "checkout, or set SATAI_REPO_ROOT. In a notebook:\n"
        "    import os; os.environ['SATAI_REPO_ROOT'] = '/content/SAT-AI'"
    )


#: Resolved from this file, which is always importable once the package is.
#: Scripts that cannot import the package yet use `find_repo_root()` instead.
REPO_ROOT: Path = Path(__file__).resolve().parent.parent
