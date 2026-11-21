"""Affordance Calculus & Assumption Attribution — Reproducibility Package."""
from __future__ import annotations

from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = REPO_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PRECOMPUTED_DIR: Path = DATA_DIR / "precomputed"

OUTPUT_DIR: Path = REPO_ROOT / "outputs"
RESULTS_OUT_DIR: Path = OUTPUT_DIR / "results"
TABLES_OUT_DIR: Path = OUTPUT_DIR / "tables"
SOURCE_DATA_OUT_DIR: Path = OUTPUT_DIR / "source_data"
FIGURES_OUT_DIR: Path = OUTPUT_DIR / "figures"


def ensure_output_dirs(base_output_dir: Path | None = None) -> dict[str, Path]:
    """Create and return output subdirectories on demand."""
    out = Path(base_output_dir) if base_output_dir is not None else OUTPUT_DIR
    dirs = {
        "root": out,
        "results": out / "results",
        "tables": out / "tables",
        "source_data": out / "source_data",
        "figures": out / "figures",
        "shards": out / "results" / "shards",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs
