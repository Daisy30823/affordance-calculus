# An Affordance Calculus Attributes Evolutionary Outcomes to Modelling Assumptions

[![Python 3.11–3.13](https://img.shields.io/badge/python-3.11%E2%80%933.13-blue.svg)](#system-requirements)
[![Manuscript Status](https://img.shields.io/badge/status-Manuscript%20in%20preparation%20%2F%20submitted-orange.svg)](#citation--manuscript-status)

Standalone, minimal, self-contained code and data reproducibility package accompanying the manuscript:

> **An affordance calculus attributes evolutionary outcomes to modelling assumptions**  
> *Manuscript prepared for submission to Nature Computational Science.*

---

## 1. Overview

Evolutionary game-theoretic models of cooperation typically fix by hand which behavioural strategies exist, and headline conclusions about costly punishment have reversed when that hand-picked strategy set changed (e.g., among published 4-, 9-, 12-, 17-, and 24-strategy optional public goods models).

This repository implements the **affordance calculus**, a general computational procedure that:
1. **Compiles explicit institutional assumptions into strategy sets**: Seven binary primitive affordances ($\alpha \in \{0,1\}^7$ — exit option `OPT`, contribution observability `OBS`, antisocial targeting `ANTI`, loner targetability `LTG`, defector agency `DAG`, loner agency `LAG`, and response stage `RET`) are compiled under a stated policy grammar (`base` or count-contingent `cnt`) into the complete, non-redundant set of behaviourally distinct strategies $S(\alpha)$ and exact group payoffs across four sanctioning mechanisms (`punish`, `reward`, `exclude`, `pool`).
2. **Evaluates exact evolutionary dynamics**: Solves the finite-population Moran process with exponential fitness $\exp(w\pi)$ in the rare-mutation limit across all $2^7 = 128$ lattice nodes (collapsing to 70 distinct games per mechanism) under four mutation kernels (`uniform`, `action`, `trait`, `redraw`) using hypergeometric co-player sampling and the subtraction-free Grassmann–Taksar–Heyman elimination algorithm in extended precision, complemented by Numba-accelerated finite-mutation individual-based Moran simulations.
3. **Attributes outcomes to assumptions**: Computes exact Shapley values, pairwise Shapley interaction indices, Harsanyi (Möbius) dividends, permission-structure (precedence-constrained) Shapley values, and Monte Carlo permutation-sampling estimators across reference parameters, 4,000 random parameter sweeps (Designs A and B), $2^{11}$-node extended lattices comparing affordances against parameter shifts, and an exploratory cross-society consistency check on the public experimental data of Herrmann, Thöni & Gächter (2008).

---

## 2. Repository Structure

```text
affordance-calculus-reproducibility/
├── README.md                          # Documentation and reproduction instructions
├── requirements.txt                   # Tested, pinned Python package versions
├── .gitignore                         # Ignores caches, virtual environments, and runtime outputs/
│
├── src/                               # Core scientific modules
│   ├── __init__.py                    # Dynamic repository-relative path resolution
│   ├── affordance.py                  # Compiler, vectorised/reference payoffs, Moran/GTH solver, Shapley/dividends
│   ├── lattice_models.py              # Validations, reference/variant lattices, parameter sweeps, extended lattices
│   ├── abm.py                         # Numba-accelerated finite-mutation individual-based Moran simulator
│   ├── htg_analysis.py                # Cross-society Kendall tau-b permutation & bootstrap analysis
│   ├── summary_stats.py               # Recomputes attributions, tables, Source Data, and 40 verification checks
│   └── plotting.py                    # Renders Main Figs. 1–5 and Extended Data Figs. 1–7 (PDF + 300-dpi PNG)
│
├── scripts/                           # Reviewer & reproduction CLI entry points
│   ├── run_demo.py                    # Quick live model demo & sanity verification (~28 s)
│   ├── reproduce_results.py           # Recomputes all statistics, tables, Source Data & verifies 40 claims (~2.5 min)
│   ├── reproduce_figures.py           # Regenerates all 12 Main & Extended Data figures (~20 s)
│   └── run_full_experiments.py        # Full from-scratch simulation pipeline (with --subset & shard support)
│
├── data/                              # Minimal canonical input & archived long-run node data (74.4 MB total)
│   ├── raw/
│   │   └── HerrmannThoeniGaechterDATA.csv     # Public CC0 cross-society experimental dataset (Dryad)
│   └── precomputed/                           # Archived node-level outputs of long-running sweeps, lattices & ABM
│
├── tests/                             # Automated pytest unit & package-hygiene test suite (17 tests)
│   ├── test_compiler_and_payoffs.py
│   ├── test_dynamics_and_kernels.py
│   ├── test_attribution_and_abm.py
│   └── test_reproducibility_package.py
│
└── docs/                              # Reproducibility, mapping & cleanup audit documentation
    ├── paper_to_code_mapping.md       # Item-by-item mapping of every figure panel, table, and claim
    ├── code_cleanup_report.md         # File-by-file cleanup, deduplication, and read-only integrity audit
    └── reproducibility_report.md      # Full benchmark timings, numerical comparison tables & NCS checklist details
```

All runtime outputs are written automatically on demand to `outputs/` (`outputs/figures/`, `outputs/tables/`, `outputs/source_data/`, `outputs/results/`), resolved dynamically relative to the repository root so scripts can be invoked from any working directory.

---

## 3. System Requirements

### Software Dependencies
- **Python**: `>= 3.11, < 3.14`
- **Scientific Python stack** (exact versions verified in clean-room testing in `requirements.txt`):
  - `numpy==2.5.5` (also tested with `2.4.6`)
  - `scipy==1.18.1` (also tested with `1.17.1`)
  - `pandas==3.1.1` (also tested with `2.3.3`)
  - `matplotlib==3.11.2` (also tested with `3.11.1`)
  - `statsmodels==0.15.0`
  - `numba==0.68.1` (also tested with `0.67.0`)
  - `mpmath==1.4.1` (also tested with `1.3.0`)
  - `pytest==9.2.0`

### Tested Operating Systems
- **Windows 11 (x86_64)** — Windows-11-10.0.26200-SP0 with Python 3.12.13 (clean-environment verification) and Python 3.11.16 (original manuscript computation).
- Pure cross-platform Python (`pathlib` dynamic path resolution; no OS-specific shell or compiler calls), compatible with standard **Linux (x86_64 / ARM64)** and **macOS** environments.

### Hardware Requirements
- **No GPU or specialised hardware required.**
- **Quick Demo, Statistical Verification & Figure Reproduction**: Runs on any standard desktop or laptop CPU with **$\ge$ 4 GB RAM** in under 3 minutes.
- **Full From-Scratch Reproduction** (all 4,000 parameter-sweep draws, $2^{11}$ extended lattices, and $48 \times 10^7$-step ABM trajectories): Standard desktop CPU works via shard-based independent processes; a multi-core workstation (e.g., 14–20 logical threads, 16–32 GB RAM) is recommended to parallelise shards.

---

## 4. Installation Guide

From a terminal, unpack or clone the repository, create an isolated virtual environment, and install the pinned dependencies:

```bash
cd affordance-calculus-reproducibility

# Create and activate a clean virtual environment
python -m venv .venv
# On Linux / macOS:
source .venv/bin/activate
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Install exact dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

- **Typical installation time on a normal desktop computer**: **~3–4 minutes** (measured wall-clock time: **206 seconds / 3.4 minutes** from scratch including virtual-environment creation and binary wheel download/installation on a standard desktop connection).

---

## 5. Demo & Quick Start (Primary Reviewer Entry Point)

To verify the environment, run core theoretical boundary checks, and execute a **100% live from-scratch demonstration** of the compiler, exact Markov chain solver, Shapley/interaction attribution, finite-mutation ABM, and raw cross-society empirical data loader:

```bash
python scripts/run_demo.py
```

- **Expected run time on a normal desktop computer**: **~25–30 seconds** (measured: **28.1 seconds**, including first-call Numba JIT compilation).
- **Expected output**:
  1. Confirms platform and dependency versions and passes all **8/8** directed payoff/grammar boundary tests.
  2. **Live model compilation & solve (`Mode a`)**: Compiles all 4 mechanism families (confirming **70 distinct games** per family) and solves all $128 \times 4 = 512$ peer-punishment Markov chains live (~0.2 s), printing the exact stationary cooperation, antisocial punisher frequency, loner frequency, and relaxation time for the published **4-, 9-, 12-, 17-, and 24-strategy sets** (`0.913`, `0.913`, `0.627`, `0.466`, `0.321`) and exact dispute Shapley allocations across all 4 mutation kernels (`uniform`: `LTG -0.248`, `DAG -0.160`, `LAG -0.113`, `ANTI -0.070`, `I(DAG,LTG) = -0.380`).
  3. **Live finite-mutation ABM (`Mode a`)**: Runs $2 \times 10^5$ Moran birth–death steps at $\mu = 0.005$ on the 4- and 24-strategy nodes.
  4. **Live empirical check (`Mode a`)**: Aggregates the 16 participant pools directly from `data/raw/HerrmannThoeniGaechterDATA.csv` and reports Kendall $\tau_b$ for P1 (`-0.477`) and P4 (`-0.583`).

---

## 6. Reproducing Paper Figures and Tables

We explicitly distinguish between **(a) live model simulation** (`run_demo.py` and `run_full_experiments.py`), **(b) recomputing attributions, summary statistics, and tables from archived node-level simulation outputs** (`reproduce_results.py --quick`), and **(c) rendering figures from canonical node data and recomputed summaries** (`reproduce_figures.py`):

### Step 1: Recompute All Manuscript Statistics, Tables & Source Data (`Mode b`)
```bash
python scripts/reproduce_results.py --quick
```
- **Expected run time**: **~2.5 minutes** (measured: **149 seconds**).
- **What it computes**: Reads the archived node-level simulation outputs in `data/precomputed/`, freshly recomputes all **72,000+ parameter-sweep Shapley & conditional Shapley attributions** (`outputs/results/sweepA_attrib.csv`, `sweepB_attrib.csv`), selection-scan crossings (`selection_scan.csv`), **288,800-permutation Monte Carlo Shapley validation** (`mc_shapley.csv`), full-precision `outputs/results/paper_numbers.json`, **Main Table 1 + Supplementary Tables 1–13** (`outputs/tables/*.csv`, 21 CSV tables), and **all 17 figure Source Data files** (`outputs/source_data/`), then executes **40 automated verification assertions** against the manuscript's reported numbers.

### Step 2: Regenerate All Main & Extended Data Figures (`Mode c`)
```bash
python scripts/reproduce_figures.py
```
- **Expected run time**: **~20 seconds** after Step 1 (measured: **19.5 seconds**; if run standalone before Step 1, it automatically computes summary statistics first in ~2.8 minutes total).
- **Outputs** (`outputs/figures/`, both vector `.pdf` and 300-dpi `.png`):
  - **Main Figures**: `Fig1` (pipeline, compiled sizes, strong-selection $1/M$ scaling), `Fig2` (128-node peer-punishment cooperation & antisocial maps), `Fig3` (4-vs-24 Shapley waterfall, Design-B violins, interaction matrix), `Fig4` (4 mutation kernels at $w=1$ & $w=0$, Design-B most-negative shares, $w$-scan), `Fig5` (cross-mechanism conditional heatmap & $2^9$ affordances vs. parameters).
  - **Extended Data Figures**: `ED_Fig1` ($w$-scan cooperation & Shapley curves), `ED_Fig2` (antisocial-capable conditional Shapley heatmap), `ED_Fig3` (MC Shapley $1/\sqrt{m}$ RMSE & 95% CI coverage), `ED_Fig4` (128-node maps for reward, ostracism & pool punishment), `ED_Fig5` (cross-society check P1, P4, P5 & Kendall $\tau$ CIs), `ED_Fig6` (finite-mutation ABM vs. rare-mutation limit), `ED_Fig7` (threshold grammar, $n=4$ & void ostracism variants).

See [`docs/paper_to_code_mapping.md`](docs/paper_to_code_mapping.md) for the complete panel-by-panel and table-by-table mapping.

---

## 7. Full Reproduction from Scratch

To re-run the entire simulation pipeline from scratch without reading `data/precomputed/` (preserving 100% of formal parameters, sample sizes, $10^7$ ABM steps, $10^6$ permutations, and fixed random seeds):

### Fast End-to-End Connectivity & Seed-Determinism Verification (`--subset`)
```bash
python scripts/run_full_experiments.py --subset
```
- **Expected run time**: **~2 minutes** (measured: **124 seconds**). Runs every stage (`core`, `core_redraw`, `sweeps` A & B slice, `extended` $r=4$ & $r=5$ slice, `abm` slice, `htg`, `pool_convention`, `extra_checks`) into `outputs/subset_run/` and automatically verifies machine-precision agreement (`0.0` to `2.6e-16`) against `data/precomputed/`.

### Full Production Scale (Exact Manuscript Scale)
```bash
# Single-command sequential execution (or run sharded stages in parallel across CPU cores):
python scripts/run_full_experiments.py --stage all
```
For multi-core workstations (e.g., $K = 20$ independent worker processes, matching the original 20-core workstation run):
```bash
python scripts/run_full_experiments.py --stage core
# Launch shards i = 0 .. 19 in parallel across cores:
python scripts/run_full_experiments.py --stage sweeps   --shard <i> --num-shards 20
python scripts/run_full_experiments.py --stage extended --shard <i> --num-shards 20
python scripts/run_full_experiments.py --stage abm      --shard <i> --num-shards 8
# Merge shards (asserts 100% shard completeness), run empirical tests & build all outputs:
python scripts/run_full_experiments.py --stage merge --num-shards 20
python scripts/run_full_experiments.py --stage htg
python scripts/run_full_experiments.py --stage extra
python scripts/run_full_experiments.py --stage summarize
```
- **Random seeds**: Design A `SeedSequence(20261109)` (2,000 draws), Design B `SeedSequence(20261110)` (2,000 draws), ABM simulation seeds `1` and `2` (`1000*seed + int(mu*1e4)`), kernel validation seed `7`, MC Shapley seed `11`, HTG permutation/bootstrap seeds `1–4`.
- **Estimated full-scale compute cost**: ~1.5–2.5 hours wall-clock across 20 CPU threads (~25–35 CPU-core hours total, dominated by the 4,000 parameter-sweep draws across 4 families, 4 kernels, and threshold/void variants, plus the $48 \times 10^7$-step ABM trajectories and 615,870 exhaustive two-type group payoff comparisons); `< 4 GB` RAM per shard.

---

## 8. Data Description

Total size of `data/` is **74.36 MB** (largest single file: `29.21 MB`, well within GitHub's 50 MB / 100 MB limits):

| Path | Format | Description & Corresponding Manuscript Items |
| :--- | :--- | :--- |
| `data/raw/HerrmannThoeniGaechterDATA.csv` | CSV (5.81 MB) | Public cross-society public-goods punishment experiment data of Herrmann, Thöni & Gächter (2008), Dryad `doi:10.5061/dryad.87301` (CC0). 16 pools, 1,120 P-experiment subjects (`ED Fig. 5`, `SI Table 9`). |
| `data/precomputed/lattice_reference.csv`, `lattice_reference_redraw.csv` | CSV (1.20 MB) | Node-level stationary outcomes (`coop`, `defect`, `loner`, `anti`, `anti_only`, `pro`, `payoff`, `relax`, `nstrat`, `digest`) for `ref`, `neutral` ($w=0$), `ref_cnt`, `ref_void`, and `ref_n4` across 4 mutation kernels (`Table 1`, `Figs. 1–4`, `ED Figs. 4, 7`, `SI Tables 2–5, 12`). |
| `data/precomputed/lattice_pool_convention.csv`, `pool_convention.json` | CSV / JSON | Pool-punishment fee convention sensitivity (`always` vs. `ifgame`; `Results`, `SI Table 2`). |
| `data/precomputed/selection_scan_nodes.csv`, `selection_scan_nodes_redraw.csv` | CSV (1.37 MB) | Node-level outcomes across 41 log-spaced selection intensities $w \in [10^{-3}, 10^2]$ for all 4 kernels (`Fig. 4c`, `ED Fig. 1`). |
| `data/precomputed/sweepA_params.csv`, `sweepA_nodes.npz`, `sweepA_nodes_redraw.npz` | CSV / NPZ (32.58 MB) | Sampled parameters and node-level `(idx, fam, var, ker, node, coop, anti)` arrays for Design A (2,000 sets, seed `20261109`; `Results`, `SI Table 6`). |
| `data/precomputed/sweepB_params.csv`, `sweepB_nodes.npz`, `sweepB_nodes_redraw.npz` | CSV / NPZ (33.13 MB) | Sampled parameters and node-level outcome arrays for Design B (2,000 sets, seed `20261110`; `Figs. 3b, 4b, 5a`, `ED Fig. 2`, `SI Tables 4, 6`). |
| `data/precomputed/extended_lattice.csv`, `extended_lattice_r5_boundary.csv` | CSV (0.19 MB) | Exact outcomes for all $2^{11} = 2,048$ nodes of the extended peer-punishment lattice ($r=4$ vs. $3$ in `Fig. 5b` & `SI Table 7`; boundary $r=5=n$ in `SI Table 8`; MC Shapley benchmark in `ED Fig. 3` & `SI Table 11`). |
| `data/precomputed/abm_finite_mutation.csv` | CSV (8 KB) | Individual-based Moran simulation cooperation rates across 16 dispute nodes, $\mu \in \{0.001, 0.005, 0.02\}$, 2 seeds, $10^7$ events (`ED Fig. 6`, `SI Table 10`). |
| `data/precomputed/htg_pool.csv`, `htg_country.csv`, `htg_tests.json` | CSV / JSON | Pool-level (16 pools) and country-level (15 countries) statistics, $10^6$-permutation Kendall $\tau_b$ $P$ values, Holm corrections, and 20,000-rep bootstrap CIs (`ED Fig. 5`, `SI Table 9`). |
| `data/precomputed/validation.json`, `extra_checks.json` | JSON (13 KB) | Strong-selection analytical limit errors ($M$ up to $10^4$), finite-$M$ GT9 errors, 60-digit `mpmath` differences, 8 directed payoff checks, 280-game digest identity check, and 615,870 exhaustive two-type group payoff comparisons (`Fig. 1c`, `SI Notes 1–2`, `SI Tables 1, 13`). |

---

## 9. Testing & Validation

Run the automated unit and self-containment test suite with `pytest`:

```bash
pytest tests/ -v
```

- **Expected run time**: **~4–5 seconds** (17 tests covering Methods Eq. 1 strategy cardinality across all 512 family-profile pairs, lattice digest payoff invariance, vectorised vs. member-by-member group payoffs across all conventions/grammars, neutral $w=0$ exact $-1/6$ benchmark, tiny-probability exact formula down to $10^{-183}$, 60-digit `mpmath` stationary solver agreement $< 2\times 10^{-16}$, Shapley axioms & Harsanyi dividend recovery, precedence-constrained Shapley efficiency, ABM seed determinism, GitHub file-size compliance, and absence of hardcoded paths).

---

## 10. Citation & Manuscript Status

This code accompanies a manuscript **in preparation / submitted** to *Nature Computational Science* (not yet published; no DOI, volume, or page numbers assigned yet):

```bibtex
@unpublished{affordance_calculus_manuscript,
  title  = {An affordance calculus attributes evolutionary outcomes to modelling assumptions},
  author = {[Authors — to be updated upon publication]},
  note   = {Manuscript prepared for submission to Nature Computational Science (code and data reproducibility archive)},
  year   = {2026}
}
```

When reusing the cross-society experimental dataset in `data/raw/HerrmannThoeniGaechterDATA.csv`, please also cite:
- Herrmann, B., Thöni, C. & Gächter, S. Antisocial punishment across societies. *Science* **319**, 1362–1367 (2008).
- Herrmann, B., Thöni, C. & Gächter, S. Data from: Antisocial punishment across societies. *Dryad* https://doi.org/10.5061/dryad.87301 (2017) [CC0 1.0 Public Domain Dedication].

---

## 11. License & Availability

- **Empirical Data (`data/raw/HerrmannThoeniGaechterDATA.csv`)**: Public domain under **CC0 1.0** (Dryad `doi:10.5061/dryad.87301`).
- **Source Code & Generated Model Data (`src/`, `scripts/`, `tests/`, `data/precomputed/`)**:
  > **[LICENSE PLACEHOLDER — ACTION REQUIRED BY AUTHORS BEFORE PUBLIC RELEASE / ACCEPTANCE]**  
  > An explicit open-source software license (recommended by *Nature Computational Science* / Springer Nature policy: OSI-approved permissive license such as **MIT**, **BSD-3-Clause**, or **Apache-2.0**) has not yet been formally designated in the original submission files. No `LICENSE` file has been auto-generated to avoid assuming legal authorization on behalf of the copyright holders. Before public release on GitHub / Zenodo, the authors should add a root `LICENSE` file and insert the permanent Zenodo archive DOI in the manuscript's *Code availability* and *Data availability* statements.
