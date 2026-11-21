#!/usr/bin/env python3
"""Regenerate all Main Figures (Fig. 1–5) and Extended Data Figures (ED Fig. 1–7).

Execution transparency (Nature Computational Science Code Checklist):
  - [Mode c — Figure Rendering from Canonical Node Data & Recomputed Summaries]
    Reads node-level outputs (`data/precomputed/` by default, or `--input-dir`)
    and recomputed summary attributions (`outputs/results/`), automatically
    computing summary attributions first if they do not yet exist, and renders
    all 12 figures in vector PDF and 300-dpi PNG formats into `outputs/figures/`.

Usage (from any working directory):
    python scripts/reproduce_figures.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate Main Figures 1–5 and Extended Data Figures 1–7.")
    parser.add_argument("--input-dir", type=Path, default=None,
                        help="Directory of node-level model outputs (defaults to data/precomputed).")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Base output directory (defaults to outputs/).")
    args = parser.parse_args()

    from src import PRECOMPUTED_DIR, ensure_output_dirs, plotting

    inp_dir = args.input_dir.resolve() if args.input_dir is not None else PRECOMPUTED_DIR
    dirs = ensure_output_dirs(args.output_dir.resolve() if args.output_dir is not None else None)

    print("=" * 84)
    print("AFFORDANCE CALCULUS — FIGURE REPRODUCTION (MAIN FIGS 1–5 & EXTENDED DATA FIGS 1–7)")
    print("=" * 84)
    print(f"[Execution Mode c] Rendering figures from node-level data ({inp_dir.name})")
    print(f"                   into `{dirs['figures']}` (vector PDF + 300-dpi PNG).")

    t0 = time.perf_counter()
    files = plotting.render_all_figures(
        input_dir=inp_dir,
        results_dir=dirs["results"],
        figures_dir=dirs["figures"],
    )
    elapsed = time.perf_counter() - t0

    expected_stems = [f"Fig{i}" for i in range(1, 6)] + [f"ED_Fig{i}" for i in range(1, 8)]
    ok = True
    print("\nGenerated files:")
    for stem in expected_stems:
        pdf_f = dirs["figures"] / f"{stem}.pdf"
        png_f = dirs["figures"] / f"{stem}.png"
        pdf_ok = pdf_f.exists() and pdf_f.stat().st_size > 1024
        png_ok = png_f.exists() and png_f.stat().st_size > 1024
        ok &= (pdf_ok and png_ok)
        print(f"  [{'OK' if (pdf_ok and png_ok) else 'MISSING'}] {stem:8s} -> "
              f"{pdf_f.name:12s} ({pdf_f.stat().st_size / 1024:6.1f} KB) | "
              f"{png_f.name:12s} ({png_f.stat().st_size / 1024:6.1f} KB)")

    print("-" * 84)
    print(f"Rendered {len(files)} figure files (12 PDF + 12 PNG) in {elapsed:.2f}s.")
    print("=" * 84)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
