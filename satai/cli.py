"""Small command-line entry points for SAT-AI."""

from __future__ import annotations

import sys


def check_env_entrypoint() -> int:
    """Console-script wrapper around ``scripts/check_env.py``."""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    script = root / "scripts" / "check_env.py"
    sys.path.insert(0, str(script.parent))
    import check_env  # type: ignore[import-not-found]

    result: int = check_env.main()
    return result


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(check_env_entrypoint())
