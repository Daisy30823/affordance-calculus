#!/usr/bin/env python3
"""Recompute all manuscript & SI statistics, tables, and Source Data, and verify
every quantitative and qualitative claim against the manuscript.

Execution transparency (Nature Computational Science Code Checklist):
  - `--quick` (default): [Mode b — Recomputing Statistics from Precomputed Node Outputs]
    Reads the archived node-level model outputs in `data/precomputed/` (lattices,
    2,000-draw parameter sweeps A & B, selection scans, 2^11 extended lattices,
    10^7-step finite-mutation ABM runs, and 10^6-permutation HTG tests), freshly
    recomputes all 72,000+ parameter-sweep Shapley attributions, 288,800-permutation
    Monte Carlo Shapley validation, `outputs/results/paper_numbers.json`, Main Table 1
    & SI Tables 1–13 (`outputs/tables/`), and all 17 figure Source Data CSVs
    (`outputs/source_data/`), then runs 33 automated verification checks.
  - `--input-dir <dir>`: Runs the exact same recomputation and verification
    against freshly simulated node-level outputs (e.g. `outputs/results` from
    `scripts/run_full_experiments.py`).

Usage (from any working directory):
    python scripts/reproduce_results.py --quick
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def print_manuscript_comparison(N: dict) -> None:
    u = N["ref"]["ref"]["punish:uniform"]
    sh = u["share"]
    rows = [
        ("Abstract / Results: Distinct games per family", "70", f"{N['distinct_games']['punish']}"),
        ("Table 1 / Abstract: Cooperation at 4-strategy node (H4)", "0.913 (0.91)", f"{N['named']['H4']['coop']:.3f}"),
        ("Table 1: Cooperation at 9-strategy node (GT9)", "0.913", f"{N['named']['GT9']['coop']:.3f}"),
        ("Table 1: Cooperation at 12-strategy node (GT12)", "0.627", f"{N['named']['GT12']['coop']:.3f}"),
        ("Table 1: Cooperation at 17-strategy node (GT17)", "0.466", f"{N['named']['GT17']['coop']:.3f}"),
        ("Table 1 / Abstract: Cooperation at 24-strategy node (RN24)", "0.321 (0.32)", f"{N['named']['RN24']['coop']:.3f}"),
        ("Results: Total cooperation drop (H4 -> RN24)", "0.591", f"{N['drop_4_24']:.3f}"),
        ("Abstract / Fig 3a: phi(LTG) & share of drop", "-0.248 (42%)", f"{u['phi']['LTG']:+.3f} ({100*sh['LTG']:.0f}%)"),
        ("Results / Fig 3a: phi(DAG) & share of drop", "-0.160 (27%)", f"{u['phi']['DAG']:+.3f} ({100*sh['DAG']:.0f}%)"),
        ("Results / Fig 3a: phi(LAG) & share of drop", "-0.113 (19%)", f"{u['phi']['LAG']:+.3f} ({100*sh['LAG']:.0f}%)"),
        ("Abstract / Fig 3a: phi(ANTI) & share of drop", "-0.070 (12%)", f"{u['phi']['ANTI']:+.3f} ({100*sh['ANTI']:.0f}%)"),
        ("Results / Fig 3c: Interaction I(DAG, LTG)", "-0.38", f"{u['I']['DAGxLTG']:+.2f}"),
        ("Results / SI Table 3: Harsanyi dividend d({DAG, LTG})", "-0.52", f"{u['dividends']['0110']:+.2f}"),
        ("Results / Fig 3b: LTG most negative in Design B (overall / w>=1)", "59% / 91%",
         f"{100*N['sweepB']['punish:0:uniform']['mostneg']['LTG']:.0f}% / {100*N['sweepB']['punish:0:uniform']['mostneg_LTG_by_w']['wge1']:.0f}%"),
        ("Results / Fig 4a: Trait-reset kernel phi(DAG) / phi(LTG)", "-0.127 / +0.042",
         f"{N['ref']['ref']['punish:trait']['phi']['DAG']:+.3f} / {N['ref']['ref']['punish:trait']['phi']['LTG']:+.3f}"),
        ("Results / Fig 4a: Redraw kernel phi(LAG) / phi(LTG)", "-0.201 / -0.167",
         f"{N['ref']['ref']['punish:redraw']['phi']['LAG']:+.3f} / {N['ref']['ref']['punish:redraw']['phi']['LTG']:+.3f}"),
        ("Results: Ostracism conditional phi(ANTI) / Design B mostneg", "-0.653 / 86%",
         f"{N['ref']['ref']['exclude:uniform']['cond_phi']['ANTI']:+.3f} / {100*N['sweepB']['exclude:0:uniform']['mostneg']['ANTI']:.0f}%"),
        ("Results / Fig 5b: Extended 2^9 subcube phi(LTG) vs phi(MHI)", "-0.202 vs +0.034",
         f"{N['ext9_phi']['LTG']:+.3f} vs {N['ext9_phi']['MHI']:+.3f}"),
        ("Results / ED Fig 3: MC Shapley RMSE (1e3 / 1e4 perms)", "0.0045 / 0.0014",
         f"{N['mc'][1000]['rmse']:.4f} / {N['mc'][10000]['rmse']:.4f}"),
        ("Results / ED Fig 5: HTG P1 tau (P_Holm) & P4 tau (P_Holm)", "-0.48 (0.015) / -0.58 (0.0022)",
         f"{N['htg']['P1_AS1_RoL']['tau']:+.2f} ({N['htg']['P1_AS1_RoL']['p_holm']:.3f}) / {N['htg']['P4_benefit_ASall']['tau']:+.2f} ({N['htg']['P4_benefit_ASall']['p_holm']:.4f})"),
    ]
    print("\n" + "=" * 92)
    print(f"{'Manuscript Metric / Claim':56s} | {'Paper Value':15s} | {'Recomputed':15s}")
    print("-" * 92)
    for label, rep, got in rows:
        print(f"{label:56s} | {rep:15s} | {got:15s}")
    print("=" * 92)


def main() -> int:
    parser = argparse.ArgumentParser(description="Recompute and verify all manuscript statistics and tables.")
    parser.add_argument("--quick", action="store_true", default=True,
                        help="Recompute attributions, tables, and Source Data from archived node-level outputs (default).")
    parser.add_argument("--input-dir", type=Path, default=None,
                        help="Directory of node-level model outputs (defaults to data/precomputed).")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Base output directory (defaults to outputs/).")
    args = parser.parse_args()

    from src import PRECOMPUTED_DIR, ensure_output_dirs, summary_stats

    inp_dir = args.input_dir.resolve() if args.input_dir is not None else PRECOMPUTED_DIR
    dirs = ensure_output_dirs(args.output_dir.resolve() if args.output_dir is not None else None)

    print("=" * 92)
    print("AFFORDANCE CALCULUS — STATISTICAL RECOMPUTATION & MANUSCRIPT VERIFICATION")
    print("=" * 92)
    if inp_dir == PRECOMPUTED_DIR:
        print("[Execution Mode b] Reading archived node-level simulation outputs from `data/precomputed/`")
        print("                   and freshly recomputing all Shapley attributions, quantiles, Wilson CIs,")
        print("                   Monte Carlo Shapley benchmarks, tables, and figure Source Data.")
    else:
        print(f"[Execution Mode] Recomputing all statistics from node-level outputs in: {inp_dir}")

    t0 = time.perf_counter()
    N = summary_stats.compute_paper_numbers(input_dir=inp_dir, results_out_dir=dirs["results"])
    t_num = time.perf_counter() - t0
    print(f"\n[1/3] Recomputed `paper_numbers.json`, sweep attributions (A & B), w-scan, and MC Shapley ({t_num:.2f}s)")

    t1 = time.perf_counter()
    tables = summary_stats.export_paper_tables(input_dir=inp_dir, results_out_dir=dirs["results"], tables_dir=dirs["tables"])
    sdata = summary_stats.export_source_data(input_dir=inp_dir, results_out_dir=dirs["results"], source_data_dir=dirs["source_data"])
    t_exp = time.perf_counter() - t1
    print(f"[2/3] Exported {len(tables)} manuscript/SI tables to `outputs/tables/` and {len(sdata)} Source Data files to `outputs/source_data/` ({t_exp:.2f}s)")

    print_manuscript_comparison(N)

    print("\n[3/3] Running automated quantitative & qualitative verification suite:")
    checks = summary_stats.verify_manuscript_claims(N)
    n_pass = sum(c["passed"] for c in checks)
    for c in checks:
        status = "PASS" if c["passed"] else "FAIL"
        suffix = f"  [{c['detail']}]" if c["detail"] else ""
        print(f"  [{status}] {c['name']}{suffix}")

    total_s = time.perf_counter() - t0
    print("-" * 92)
    print(f"VERIFICATION SUMMARY: {n_pass}/{len(checks)} checks passed in {total_s:.2f}s.")
    print("=" * 92)
    return 0 if n_pass == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
