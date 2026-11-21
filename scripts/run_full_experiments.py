#!/usr/bin/env python3
"""Full from-scratch simulation pipeline (Tier 2 — Full Reproduction).

Re-runs all theoretical validations, reference & variant lattices, selection
scans, parameter sweeps (Designs A & B, 2,000 draws each with seeds 20261109
and 20261110), 2^11 extended lattices (r=4 social dilemma and r=5 boundary),
finite-mutation Moran ABM simulations (10^7 birth-death events per run),
cross-society permutation tests (10^6 permutations), pool-fee convention and
exhaustive payoff checks from scratch.

Supports two execution modes:
  1. Full production scale (default, exact manuscript settings; supports shard
     parallelisation via `--shard i --num-shards k`):
       python scripts/run_full_experiments.py --stage all
  2. End-to-end connectivity & seed-reproducibility verification (`--subset`):
     Executes every single from-scratch simulation stage on a small deterministic
     slice (using the exact production SeedSequence states and RNG seeds) in
     ~15–25 seconds and verifies numerical agreement against `data/precomputed/`:
       python scripts/run_full_experiments.py --subset
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def verify_subset_against_archive(sub_out: Path, precomp_dir: Path) -> bool:
    """Verify that from-scratch `--subset` outputs match archived `data/precomputed/`
    at machine / statistical precision using identical random seeds."""
    ok = True
    print("\nVerifying from-scratch `--subset` outputs against archived `data/precomputed/`:")

    # 1. Reference lattice exact match
    lat_new = pd.read_csv(sub_out / "lattice_reference.csv", dtype={"node": str}).set_index(["tag", "family", "kernel", "node"])
    lat_ref = pd.read_csv(precomp_dir / "lattice_reference.csv", dtype={"node": str}).set_index(["tag", "family", "kernel", "node"])
    diff_lat = float((lat_new["coop"] - lat_ref.loc[lat_new.index, "coop"]).abs().max())
    c1 = diff_lat < 1e-12
    ok &= c1
    print(f"  [{'PASS' if c1 else 'FAIL'}] lattice_reference.csv max |coop diff| = {diff_lat:.3e}")

    # 2. Redraw lattice exact match
    latr_new = pd.read_csv(sub_out / "lattice_reference_redraw.csv", dtype={"node": str}).set_index(["tag", "family", "kernel", "node"])
    latr_ref = pd.read_csv(precomp_dir / "lattice_reference_redraw.csv", dtype={"node": str}).set_index(["tag", "family", "kernel", "node"])
    diff_latr = float((latr_new["coop"] - latr_ref.loc[latr_new.index, "coop"]).abs().max())
    c2 = diff_latr < 1e-12
    ok &= c2
    print(f"  [{'PASS' if c2 else 'FAIL'}] lattice_reference_redraw.csv max |coop diff| = {diff_latr:.3e}")

    # 3. Sweep A & B seed-deterministic parameter draws & node outcomes
    for design in ["A", "B"]:
        p_new = pd.read_csv(sub_out / f"sweep{design}_params.csv").sort_values("idx")
        p_ref = pd.read_csv(precomp_dir / f"sweep{design}_params.csv").sort_values("idx").iloc[:len(p_new)]
        p_diff = float((p_new[["r", "sigma", "gamma", "beta", "w"]].values - p_ref[["r", "sigma", "gamma", "beta", "w"]].values).max())
        r_new = np.load(sub_out / f"sweep{design}_nodes.npz")["recs"]
        r_ref = np.load(precomp_dir / f"sweep{design}_nodes.npz")["recs"]
        r_ref_sub = r_ref[np.isin(r_ref[:, 0], p_new["idx"].values)]
        # Sort both by (idx, fam, var, ker, node)
        s_new = r_new[np.lexsort((r_new[:, 4], r_new[:, 3], r_new[:, 2], r_new[:, 1], r_new[:, 0]))]
        s_ref = r_ref_sub[np.lexsort((r_ref_sub[:, 4], r_ref_sub[:, 3], r_ref_sub[:, 2], r_ref_sub[:, 1], r_ref_sub[:, 0]))]
        c_diff = float(np.abs(s_new[:, 5:] - s_ref[:, 5:]).max())
        cond = (abs(p_diff) < 1e-14) and (c_diff < 1e-12)
        ok &= cond
        print(f"  [{'PASS' if cond else 'FAIL'}] sweep{design} ({len(p_new)} draws, seed {20261109 if design=='A' else 20261110}): "
              f"param diff = {abs(p_diff):.3e}, outcome max diff = {c_diff:.3e}")

    # 4. Extended lattice subset match
    for fname in ["extended_lattice.csv", "extended_lattice_r5_boundary.csv"]:
        ex_new = pd.read_csv(sub_out / fname, dtype={"node": str}).set_index("node")
        ex_ref = pd.read_csv(precomp_dir / fname, dtype={"node": str}).set_index("node")
        ex_diff = float((ex_new["coop"] - ex_ref.loc[ex_new.index, "coop"]).abs().max())
        cond = ex_diff < 1e-12
        ok &= cond
        print(f"  [{'PASS' if cond else 'FAIL'}] {fname} ({len(ex_new)} nodes) max |coop diff| = {ex_diff:.3e}")

    # 5. HTG pool table exact match
    hp_new = pd.read_csv(sub_out / "htg_pool.csv", index_col=0).sort_index()
    hp_ref = pd.read_csv(precomp_dir / "htg_pool.csv", index_col=0).sort_index()
    hp_diff = float((hp_new["AS_1"] - hp_ref["AS_1"]).abs().max())
    c5 = hp_diff < 1e-12
    ok &= c5
    print(f"  [{'PASS' if c5 else 'FAIL'}] htg_pool.csv (16 pools from raw CC0 CSV) max |AS_1 diff| = {hp_diff:.3e}")

    # 6. Pool convention exact match
    pc_new = pd.read_csv(sub_out / "lattice_pool_convention.csv", dtype={"node": str}).set_index(["tag", "node"])
    pc_ref = pd.read_csv(precomp_dir / "lattice_pool_convention.csv", dtype={"node": str}).set_index(["tag", "node"])
    pc_diff = float((pc_new["coop"] - pc_ref.loc[pc_new.index, "coop"]).abs().max())
    c6 = pc_diff < 1e-12
    ok &= c6
    print(f"  [{'PASS' if c6 else 'FAIL'}] lattice_pool_convention.csv max |coop diff| = {pc_diff:.3e}")

    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full from-scratch experiments (or --subset pipeline check).")
    parser.add_argument("--stage", choices=["all", "core", "sweeps", "extended", "abm", "htg", "extra", "merge", "summarize"],
                        default="all", help="Pipeline stage to execute (default: all).")
    parser.add_argument("--subset", action="store_true",
                        help="Run a fast deterministic end-to-end slice of every stage and verify against archived precomputed data.")
    parser.add_argument("--shard", type=int, default=0, help="Shard index i in 0..k-1 for sharded stages.")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of shards k (default: 1).")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (defaults to outputs/subset_run in --subset mode, else outputs/).")
    args = parser.parse_args()

    from src import (
        OUTPUT_DIR,
        PRECOMPUTED_DIR,
        RAW_DATA_DIR,
        abm,
        ensure_output_dirs,
        htg_analysis,
        lattice_models,
        plotting,
        summary_stats,
    )

    if args.output_dir is not None:
        base_out = args.output_dir.resolve()
    else:
        base_out = (OUTPUT_DIR / "subset_run") if args.subset else OUTPUT_DIR

    dirs = ensure_output_dirs(base_out)
    res_dir = dirs["results"]
    sh_dir = dirs["shards"]

    # Clean existing shard files in subset mode to avoid stale shard merges
    if args.subset and sh_dir.exists():
        for p in sh_dir.glob("*"):
            if p.is_file():
                p.unlink()

    t_all = time.perf_counter()
    print("=" * 88)
    if args.subset:
        print("AFFORDANCE CALCULUS — FROM-SCRATCH PIPELINE CONNECTIVITY & SEED CHECK (--subset)")
    else:
        print("AFFORDANCE CALCULUS — FULL FROM-SCRATCH EXPERIMENT PIPELINE")
    print("=" * 88)
    print(f"Output directory: {base_out}")

    n_draws = 4 if args.subset else lattice_models.N_SWEEP
    n_ext = 16 if args.subset else None
    max_M_val = 1000 if args.subset else 10000

    if args.stage in ("all", "core"):
        print("\n[Stage: core] Running validation, reference/variant lattices, and w-scans (4 kernels)...")
        lattice_models.run_core(res_dir, max_M_limit=max_M_val, verbose=True)
        lattice_models.run_core_redraw(res_dir, verbose=True)

    if args.stage in ("all", "sweeps"):
        print(f"\n[Stage: sweeps] Running parameter sweeps A & B (n_draws={n_draws}, shard {args.shard}/{args.num_shards})...")
        t0 = time.perf_counter()
        for design in ["A", "B"]:
            lattice_models.sweep_shard(design, args.shard, args.num_shards, shards_dir=sh_dir, n_draws=n_draws)
            lattice_models.sweep_redraw_shard(design, args.shard, args.num_shards, shards_dir=sh_dir, n_draws=n_draws)
        print(f"  sweep shards completed ({time.perf_counter() - t0:.2f}s)")

    if args.stage in ("all", "extended"):
        print(f"\n[Stage: extended] Running 2^11 extended lattices r=4 & r=5 (limit={n_ext or 2048}, shard {args.shard}/{args.num_shards})...")
        t0 = time.perf_counter()
        lattice_models.ext_shard(args.shard, args.num_shards, shards_dir=sh_dir, r_hi=4.0, prefix="ext", limit=n_ext)
        lattice_models.ext_shard(args.shard, args.num_shards, shards_dir=sh_dir, r_hi=5.0, prefix="ext_r5", limit=n_ext)
        print(f"  extended lattice shards completed ({time.perf_counter() - t0:.2f}s)")

    if args.stage in ("all", "abm"):
        if args.subset:
            abm_nodes = ["1100000", "1111110"]
            abm_mus = [0.005, 0.02]
            abm_seeds = [1, 2]
            t_steps, burn = 200_000, 20_000
        else:
            abm_nodes, abm_mus, abm_seeds = None, None, None
            t_steps, burn = abm.T_STEPS, abm.BURN
        n_abm_jobs = len(abm.jobs(nodes=abm_nodes, mus=abm_mus, seeds=abm_seeds))
        print(f"\n[Stage: abm] Running finite-mutation Moran ABM ({n_abm_jobs} jobs, T={t_steps:,}, shard {args.shard}/{args.num_shards})...")
        t0 = time.perf_counter()
        abm.run_abm_shard(args.shard, args.num_shards, shards_dir=sh_dir,
                          t_steps=t_steps, burn=burn, nodes=abm_nodes, mus=abm_mus, seeds=abm_seeds)
        print(f"  ABM shard completed ({time.perf_counter() - t0:.2f}s)")
    else:
        n_abm_jobs = None

    if args.stage in ("all", "merge") and args.num_shards == 1 or args.stage == "merge":
        print("\n[Stage: merge] Merging sweep, extended lattice, and ABM shards...")
        lattice_models.merge_shards(out_dir=res_dir, shards_dir=sh_dir,
                                    n_draws=n_draws, n_ext=n_ext,
                                    n_abm_jobs=n_abm_jobs if args.subset else None)

    if args.stage in ("all", "htg"):
        nperm = 5_000 if args.subset else htg_analysis.NPERM
        nperm_s = 2_000 if args.subset else htg_analysis.NPERM_S
        nboot = 1_000 if args.subset else 20_000
        print(f"\n[Stage: htg] Running cross-society empirical analysis (nperm={nperm:,}, nboot={nboot:,})...")
        t0 = time.perf_counter()
        htg_analysis.run_htg_analysis(
            data_csv=RAW_DATA_DIR / "HerrmannThoeniGaechterDATA.csv",
            out_dir=res_dir,
            nperm=nperm,
            nperm_s=nperm_s,
            nboot=nboot,
            verbose=not args.subset,
        )
        print(f"  HTG empirical analysis completed ({time.perf_counter() - t0:.2f}s)")

    if args.stage in ("all", "extra"):
        grams = ("base",) if args.subset else ("base", "cnt")
        print(f"\n[Stage: extra] Running pool-fee convention and exhaustive payoff checks (grammars={grams})...")
        t0 = time.perf_counter()
        lattice_models.run_pool_convention(out_dir=res_dir)
        lattice_models.run_extra_checks(out_dir=res_dir, grammars=grams)
        print(f"  Extra checks completed ({time.perf_counter() - t0:.2f}s)")

    ok = True
    if args.subset and args.stage == "all":
        ok = verify_subset_against_archive(res_dir, PRECOMPUTED_DIR)
    elif not args.subset and args.stage in ("all", "summarize"):
        print("\n[Stage: summarize] Computing paper_numbers.json, tables, Source Data, and figures...")
        N = summary_stats.compute_paper_numbers(input_dir=res_dir, results_out_dir=res_dir)
        summary_stats.export_paper_tables(input_dir=res_dir, results_out_dir=res_dir, tables_dir=dirs["tables"])
        summary_stats.export_source_data(input_dir=res_dir, results_out_dir=res_dir, source_data_dir=dirs["source_data"])
        plotting.render_all_figures(input_dir=res_dir, results_dir=res_dir, figures_dir=dirs["figures"])
        checks = summary_stats.verify_manuscript_claims(N)
        ok = all(c["passed"] for c in checks)

    print("-" * 88)
    print(f"PIPELINE FINISHED in {time.perf_counter() - t_all:.2f}s — Status: {'SUCCESS' if ok else 'FAILURE'}")
    print("=" * 88)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
