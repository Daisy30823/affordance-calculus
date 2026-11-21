"""Self-containment, data-integrity, and path-hygiene tests for the standalone
reproducibility package."""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import DATA_DIR, PRECOMPUTED_DIR, RAW_DATA_DIR


def test_canonical_data_files_present_and_within_github_size_limits():
    raw_file = RAW_DATA_DIR / "HerrmannThoeniGaechterDATA.csv"
    assert raw_file.exists() and raw_file.stat().st_size > 1_000_000

    required_precomputed = [
        "lattice_reference.csv",
        "lattice_reference_redraw.csv",
        "lattice_pool_convention.csv",
        "selection_scan_nodes.csv",
        "selection_scan_nodes_redraw.csv",
        "sweepA_params.csv",
        "sweepA_nodes.npz",
        "sweepA_nodes_redraw.npz",
        "sweepB_params.csv",
        "sweepB_nodes.npz",
        "sweepB_nodes_redraw.npz",
        "extended_lattice.csv",
        "extended_lattice_r5_boundary.csv",
        "abm_finite_mutation.csv",
        "htg_pool.csv",
        "htg_country.csv",
        "htg_tests.json",
        "validation.json",
        "pool_convention.json",
        "extra_checks.json",
    ]
    for fname in required_precomputed:
        fp = PRECOMPUTED_DIR / fname
        assert fp.exists(), f"Missing {fname}"
        assert 0 < fp.stat().st_size < 50 * 1024 * 1024, f"{fname} exceeds 50 MB best-practice limit"


def test_no_hardcoded_local_paths_or_cross_directory_leaks_in_code():
    """Ensure no code or test file references the original submission folder or
    any personal machine path."""
    forbidden_patterns = [
        re.compile(r"submission_NCS_affordance_calculus", re.IGNORECASE),
        re.compile(r"06_code_and_data"),
        re.compile(r"01_manuscript"),
        re.compile(r"03_cover_letter"),
        re.compile(r"04_response_to_reviewers"),
        re.compile(r"AUTHOR_TODO"),
        re.compile(r"[A-Za-z]:\\\\Users\\\\", re.IGNORECASE),
        re.compile(r"/Users/[A-Za-z0-9_]+"),
    ]
    for sub in ["src", "scripts"]:
        for py_file in (REPO_ROOT / sub).rglob("*.py"):
            txt = py_file.read_text(encoding="utf-8")
            for pat in forbidden_patterns:
                assert not pat.search(txt), f"Forbidden path pattern {pat.pattern} in {py_file.name}"
