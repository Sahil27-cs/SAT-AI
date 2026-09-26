# Setup — Windows

Written for a Windows machine with miniconda already installed. Follow in order.

---

## 1. Prerequisites

| Tool | Check | If missing |
|---|---|---|
| conda | `conda --version` | <https://docs.conda.io/projects/miniconda/> |
| git | `git --version` | <https://git-scm.com/download/win> |
| Node 20+ | `node --version` | <https://nodejs.org> (needed from Phase 14) |
| Docker Desktop | `docker --version` | <https://docker.com> (optional until Phase 16) |

Use **PowerShell**, not Command Prompt.

---

## 2. Create the environment

```powershell
cd $HOME\Documents\SAT-AI

conda env create -f environment.yml
conda activate satai
```

This takes 10–25 minutes. conda-forge is resolving a large compiled stack; that
is the cost of it working the first time (see ADR-004).

If solving is slow, use the faster solver:

```powershell
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
```

Then install the project itself:

```powershell
pip install -e ".[dev]"
```

`-e` is editable mode: edits to `satai/` take effect without reinstalling.

---

## 3. Verify

```powershell
python scripts/check_env.py
```

Expect `OK` for Phase 1 requirements and `WARN` for packages belonging to later
phases. **Paste this output when reporting a problem** — it answers most setup
questions on its own, including the RAM and GPU questions that determine where
models get trained.

Then:

```powershell
pytest -q
ruff check .
mypy
```

All three should pass on a clean checkout.

---

## 4. Environment file

```powershell
Copy-Item .env.example .env
```

Nothing needs filling in for Phase 1. Credentials become necessary in Phase 2.

`.env` is gitignored. If you ever commit it by accident, **rotate every
credential in it** — removing the file from a later commit does not remove it
from history.

---

## 5. Enable the pre-commit hooks

```powershell
pre-commit install
pre-commit run --all-files
```

These block the two expensive mistakes in this project: committing a credential,
and committing a satellite raster.

---

## 6. PyTorch and the GPU (from Phase 5)

`environment.yml` installs the **CPU** build, which is correct if you have no
NVIDIA GPU.

Check:

```powershell
nvidia-smi
```

If that reports a GPU, install the CUDA build instead — take the exact command
for your CUDA version from <https://pytorch.org/get-started/locally/>:

```powershell
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.cuda.is_available())"
```

If `nvidia-smi` is not found, that is expected and fine. U-Net training goes to
**Kaggle** (30 GPU-hours per week, free, no idle disconnect) or **Colab**;
everything else runs on CPU. The plan assumes no local GPU.

---

## 7. Troubleshooting

**`ModuleNotFoundError: No module named 'yaml'` (or `pydantic`, or `satai`)**

Your prompt says `(base)`, not `(satai)`. The environment was never created, or
was created and not activated. Check which one you are in:

```powershell
conda info --envs      # the * marks the active environment
```

Then create it if needed and activate it (§2). `python scripts\check_env.py`
reports exactly which packages are missing and which phase needs each one — run
it before anything else when an import fails.

If you need an answer in the next minute and do not want to wait for conda to
solve, the two Phase 2 scripts need only three pure-Python packages:

```powershell
pip install pydantic pydantic-settings pyyaml
pip install -e .
```

That is a stopgap, not the setup. Phase 3 needs the full conda environment for
GDAL, and installing the geospatial stack with pip on Windows is the failure
mode described below.

**`ImportError: DLL load failed while importing _gdal`**
The classic Windows GDAL failure. Almost always caused by mixing pip and conda
installs of the geospatial stack.

```powershell
conda activate satai
pip uninstall rasterio geopandas fiona pyproj shapely
conda install -c conda-forge rasterio geopandas fiona pyproj shapely --force-reinstall
```

**`conda` is not recognised**
conda is not on PATH. Use "Anaconda Prompt" from the Start menu, or run
`conda init powershell` once and reopen the terminal.

**`pip install -e .` fails with a metadata error**
Update the build tooling: `pip install --upgrade pip setuptools wheel`.

**`pytest` says `ModuleNotFoundError: No module named 'satai'`**
The package is not installed in the active environment. Confirm `satai` is
active (`conda info --envs`) and run `pip install -e ".[dev]"`.

**Environment solving takes forever or fails**
Switch to libmamba (§2). If it still fails, delete and recreate:
`conda env remove -n satai`.

**Long-path errors**
Windows' 260-character path limit bites with deeply nested node_modules. Enable
long paths, as Administrator:

```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
  -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

**Line-ending noise in every diff**
The pre-commit `mixed-line-ending` hook normalises to LF. Also set:

```powershell
git config --global core.autocrlf input
```

---

## Fallback: WSL2

If Windows setup proves intractable, WSL2 gives a Linux environment where the
geospatial stack installs without friction:

```powershell
wsl --install -d Ubuntu
```

Then follow the same steps inside Ubuntu. Keep the repository **inside** the WSL
filesystem (`~/SAT-AI`), not on `/mnt/c` — cross-filesystem I/O there is slow
enough to matter when processing thousands of tiles.
