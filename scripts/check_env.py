#!/usr/bin/env python3
"""SAT-AI environment diagnostic.

Run this first, and after every environment change::

    python scripts/check_env.py

It reports the machine's hardware, the Python environment, which scientific
packages import cleanly, and which external services are configured. It answers
the question that determines several architectural choices in this project:
*which models can be trained locally and which must go to a cloud GPU?*

Nothing here touches the network and nothing here prints a secret -- only
whether each credential is present.

Exit codes: 0 = all required checks passed, 1 = a required check failed.
"""

from __future__ import annotations

import importlib
import platform
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

GREEN, YELLOW, RED, DIM, RESET = "\033[32m", "\033[33m", "\033[31m", "\033[2m", "\033[0m"
if platform.system() == "Windows":
    # Enable ANSI on Windows 10+ terminals; fall back to no colour if it fails.
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except Exception:  # noqa: BLE001  # pragma: no cover
        GREEN = YELLOW = RED = DIM = RESET = ""

OK, WARN, FAIL = f"{GREEN}  OK  {RESET}", f"{YELLOW} WARN {RESET}", f"{RED} FAIL {RESET}"

# Packages needed at each phase. Missing Phase 1 packages are failures; later
# phases are warnings, because Phase 1 must run in a minimal environment.
REQUIRED_NOW = ["pydantic", "pydantic_settings", "yaml"]
NEEDED_LATER = [
    ("numpy", "Phase 3"),
    ("pandas", "Phase 3"),
    ("rasterio", "Phase 3 - geospatial I/O"),
    ("rioxarray", "Phase 3"),
    ("geopandas", "Phase 3 - vector"),
    ("pyproj", "Phase 3 - CRS"),
    ("skimage", "Phase 4 - Otsu baseline, co-registration"),
    ("sklearn", "Phase 4"),
    ("xgboost", "Phase 4/6 - baselines"),
    ("shap", "Phase 7 - explainability"),
    ("torch", "Phase 5 - U-Net"),
    ("segmentation_models_pytorch", "Phase 5"),
    ("onnxruntime", "Phase 9 - CPU serving"),
    ("ee", "Phase 2 - Earth Engine"),
    ("fastapi", "Phase 9"),
    ("sqlalchemy", "Phase 9"),
    # `langgraph` was listed here as a Phase 11 requirement and is not imported
    # anywhere in the project: routing is a weighted keyword scorer that must
    # work with no API key at all (requirement 39), and a graph framework buys
    # nothing at three agents and nine tools. Checking for a package nothing
    # uses trains people to ignore this script's output. See ADR-001.
    ("httpx", "Phase 9/11 - serving plane and agent HTTP"),
]


def _h(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def _row(status: str, label: str, value: str = "") -> None:
    print(f"[{status}] {label:<34} {value}")


def _human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"


def check_system() -> dict[str, object]:
    """Report OS, CPU, RAM and disk. RAM is what gates local training."""
    _h("System")
    info: dict[str, object] = {}

    _row(OK, "OS", f"{platform.system()} {platform.release()} ({platform.machine()})")
    _row(OK, "Python", f"{platform.python_version()}  [{sys.executable}]")
    info["os"] = platform.system()

    try:
        import psutil

        cores = psutil.cpu_count(logical=False) or 0
        threads = psutil.cpu_count(logical=True) or 0
        _row(OK, "CPU", f"{cores} physical / {threads} logical cores")
        info["cpu_physical"] = cores

        vm = psutil.virtual_memory()
        gb = vm.total / 1024**3
        info["ram_gb"] = round(gb, 1)
        if gb >= 15:
            _row(OK, "RAM", f"{gb:.1f} GB  (comfortable)")
        elif gb >= 7.5:
            _row(OK, "RAM", f"{gb:.1f} GB  (workable - keep tile batches small)")
        else:
            _row(
                WARN,
                "RAM",
                f"{gb:.1f} GB  (tight - lean hard on Earth Engine, reduce tile size in Phase 3)",
            )

        usage = shutil.disk_usage(REPO_ROOT)
        free_gb = usage.free / 1024**3
        info["disk_free_gb"] = round(free_gb, 1)
        status = OK if free_gb >= 60 else (WARN if free_gb >= 25 else FAIL)
        _row(
            status,
            "Free disk (repo drive)",
            f"{_human_bytes(usage.free)} free of {_human_bytes(usage.total)}",
        )
        if free_gb < 60:
            print(
                f"{DIM}       Phase 2-3 need roughly 40-60 GB for raw + processed "
                f"tiles over one AOI.{RESET}"
            )
    except ImportError:
        _row(WARN, "psutil", "not installed - cannot report CPU/RAM. pip install psutil")

    return info


def check_gpu() -> dict[str, object]:
    """Detect an NVIDIA GPU. Its absence is expected and does not block anything."""
    _h("GPU")
    info: dict[str, object] = {"gpu": None}

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            out = subprocess.run(
                [
                    nvidia_smi,
                    "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            if out.returncode == 0 and out.stdout.strip():
                for line in out.stdout.strip().splitlines():
                    _row(OK, "NVIDIA GPU", line.strip())
                info["gpu"] = out.stdout.strip().splitlines()[0].strip()
            else:
                _row(WARN, "nvidia-smi", "present but returned no device")
        except (subprocess.TimeoutExpired, OSError) as exc:
            _row(WARN, "nvidia-smi", f"failed: {exc}")
    else:
        _row(WARN, "NVIDIA GPU", "not detected")
        print(
            f"{DIM}       Expected, and fine. Plan: train the U-Net on Kaggle "
            f"(30 GPU-hours/week free) or Colab; run everything else on CPU.{RESET}"
        )

    try:
        import torch

        cuda = torch.cuda.is_available()
        _row(OK if cuda else WARN, "PyTorch CUDA", f"{torch.__version__}, cuda_available={cuda}")
        info["torch_cuda"] = cuda
    except ImportError:
        _row(WARN, "PyTorch", "not installed yet (needed from Phase 5)")

    return info


def check_packages() -> bool:
    """Import-check the scientific stack. Import, not pip list -- broken GDAL
    bindings on Windows show up as an ImportError, not a missing package."""
    _h("Python packages")
    all_required_ok = True

    for mod in REQUIRED_NOW:
        try:
            m = importlib.import_module(mod)
            _row(OK, mod, getattr(m, "__version__", ""))
        except ImportError as exc:
            _row(FAIL, mod, f"MISSING - required now ({exc})")
            all_required_ok = False

    print()
    for mod, phase in NEEDED_LATER:
        try:
            m = importlib.import_module(mod)
            _row(OK, mod, getattr(m, "__version__", "") or phase)
        except ImportError:
            _row(WARN, mod, f"not yet installed - needed at {phase}")

    return all_required_ok


def check_project() -> bool:
    """Verify the repository layout and that satai imports."""
    _h("Project")
    ok = True

    for rel in (
        "satai/config.py",
        "satai/provenance.py",
        "configs/aoi.yaml",
        "configs/risk.yaml",
        ".env.example",
        "pyproject.toml",
    ):
        path = REPO_ROOT / rel
        if path.exists():
            _row(OK, rel)
        else:
            _row(FAIL, rel, "missing")
            ok = False

    env_file = REPO_ROOT / ".env"
    if env_file.exists():
        _row(OK, ".env", "present (gitignored)")
    else:
        _row(WARN, ".env", "not created yet - copy from .env.example")

    try:
        import satai

        _row(OK, "import satai", f"v{satai.__version__}")
    except ImportError as exc:
        _row(FAIL, "import satai", f'{exc}  -> run: pip install -e ".[dev]"')
        return False

    try:
        from satai.geo.aoi import load_aoi_registry

        registry = load_aoi_registry()
        _row(
            OK, "AOI registry", f"{len(registry)} areas defined, {len(registry.selected)} selected"
        )
        for aoi in registry.aois:
            labels = "  [label source named]" if aoi.label_sources_declared else "  [none named]"
            print(
                f"{DIM}       - {aoi.id:<20} {aoi.approx_area_km2:>8,.0f} km2  "
                f"~{aoi.approx_tile_count():>5,} tiles  {aoi.utm_epsg}"
                f"{labels}{RESET}"
            )
    except Exception as exc:  # noqa: BLE001
        _row(FAIL, "AOI registry", str(exc))
        ok = False

    return ok


def check_credentials() -> None:
    """Report which services are configured. Never prints a secret value."""
    _h("External services")
    try:
        from satai.config import get_settings

        report = get_settings().capability_report()
    except Exception as exc:  # noqa: BLE001
        _row(FAIL, "settings", str(exc))
        return

    phases = {
        "llm": "Phase 11 - agents",
        "earth_engine": "Phase 2 - PRIMARY data plane",
        "cdse": "Phase 2 - secondary data plane",
        "firms": "Phase 16 - wildfire",
        "earthdata": "Phase 2 - GPM rainfall",
    }
    for service, configured in report.items():
        _row(
            OK if configured else WARN,
            service,
            "configured" if configured else f"not configured ({phases.get(service, '')})",
        )

    print(
        f"\n{DIM}None of these are required for Phase 1. Earth Engine and CDSE "
        f"become necessary in Phase 2.{RESET}"
    )


def main() -> int:
    print("=" * 74)
    print("  SAT-AI environment check")
    print("=" * 74)

    check_system()
    check_gpu()
    packages_ok = check_packages()
    project_ok = check_project()
    check_credentials()

    _h("Result")
    if packages_ok and project_ok:
        print(f"[{OK}] Phase 1 environment is good. Ready to proceed.")
        return 0
    print(f"[{FAIL}] Required checks failed - see FAIL rows above.")
    print(f'{DIM}Most common fix:  conda activate satai  &&  pip install -e ".[dev]"{RESET}')
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
