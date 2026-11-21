# Reproducibility & Verification Report

This report documents the clean-environment installation, end-to-end execution timings, unit-test results, numerical verification of manuscript and Supplementary Information claims, figure validation, working-directory isolation checks, and pre-submission technical notes for *An affordance calculus attributes evolutionary outcomes to modelling assumptions* (prepared for *Nature Computational Science*).

The timing and system specifications below can be copied directly into the **Springer Nature / Nature Computational Science Code & Software Submission Checklist**.

---

## 1. Test Environment Specifications

| Item | Specification |
| :--- | :--- |
| **Operating System** | Windows 11 (64-bit, `Windows-11-10.0.26200-SP0`, `AMD64`) |
| **Processor (CPU)** | 12th Gen Intel(R) Core(TM) i9-12900H (14 physical cores, 20 logical processors) |
| **Memory (RAM)** | 32 GB (`31.7 GB` usable physical RAM) |
| **GPU / Special Hardware** | None required (CPU-only execution) |
| **Python Interpreter** | CPython `3.12.13` (64-bit, isolated clean `venv`; original manuscript computed on CPython `3.11.16`) |
| **Installed Packages (`pip check`: 0 conflicts)** | `numpy==2.5.5`, `scipy==1.18.1`, `pandas==3.1.1`, `matplotlib==3.11.2`, `statsmodels==0.15.0`, `numba==0.68.1` (`llvmlite==0.50.0`), `mpmath==1.4.1`, `pytest==9.2.0` |

---

## 2. Measured Installation & Execution Timings (NCS Checklist Ready)

All commands were executed inside a freshly created Python virtual environment isolated from the original submission directory:

| Workflow Step | Command Executed | Execution Mode | Measured Wall-Clock Time | Exit Code |
| :--- | :--- | :--- | :--- | :--- |
| **Clean Environment & Dependency Install** | `python -m venv .venv && pip install -r requirements.txt` | Binary wheels from PyPI | **206.2 s (~3.4 min)** | `0` |
| **Automated Unit & Package Hygiene Tests** | `pytest tests/ -v` | 17 unit & self-containment tests | **4.5 s** (`3.56 s` test body) | `0` (`17/17 PASS`) |
| **Quick Live Demo (`Tier 1a`)** | `python scripts/run_demo.py` | **Mode (a)**: 100% live model compile, 512 Markov chains, live Shapley, mini ABM ($2\times 10^5$ steps) & raw CC0 empirical check | **28.1 s** (incl. ~24 s Numba JIT compile + ABM) | `0` |
| **Statistical Recomputation & Claim Verification (`Tier 1b`)** | `python scripts/reproduce_results.py --quick` | **Mode (b)**: Reads archived node-level outputs (`data/precomputed/`) and freshly recomputes 72,000+ sweep attributions, 288,800 MC Shapley permutations, 21 CSV tables, 17 Source Data files & 40 assertions | **149.1 s (~2.5 min)** | `0` (`40/40 PASS`) |
| **Main & Extended Data Figure Reproduction (`Tier 1c`)** | `python scripts/reproduce_figures.py` | **Mode (c)**: Renders `Fig1`–`Fig5` & `ED_Fig1`–`ED_Fig7` (12 vector PDFs + 12 300-dpi PNGs) | **19.5 s** | `0` (`24/24 files`) |
| **From-Scratch Pipeline Connectivity & Seed Check (`Tier 2 --subset`)** | `python scripts/run_full_experiments.py --subset` | **Mode (a)**: Runs `core`, `core_redraw`, `sweeps` A & B (4 draws, exact `SeedSequence`), `extended` ($r=4,5$), `abm`, `htg`, `pool_convention` & `extra_checks` from scratch + verifies against archive | **124.4 s (~2.1 min)** | `0` (`8/8 archive checks PASS`) |

> **Full Production-Scale From-Scratch Estimate (`python scripts/run_full_experiments.py --stage all`)**:
> Extrapolating directly from the measured shard timings on this 20-thread CPU (`core` + `core_redraw`: `38.6 s`; 4 sweep draws across A & B: `36.8 s` $\implies$ 2,000 draws $\approx 5.1$ CPU-hours sequential or **~20–30 minutes** across 20 parallel shards; $2^{11}$ extended lattices: `~3 min`; 48 ABM runs of $10^7$ steps: `~25–35 min` across 8 shards; $10^6$-permutation HTG analysis: `~4 min`; 615,870 exhaustive two-type payoff checks: `~25 s`), full from-scratch regeneration requires **~1.5 hours wall-clock on a 20-thread workstation** (~25–30 CPU-core hours sequential) and `< 4 GB` RAM per process. In the Quick mode, long-running node-level outputs (`sweepA_nodes*.npz`, `sweepB_nodes*.npz`, `abm_finite_mutation.csv`, `htg_tests.json`, `validation.json`) use the verified precomputed archives in `data/precomputed/`.

---

## 3. Unit Test Suite Results (`tests/`)

All **17 / 17** `pytest` unit tests passed:
1. `test_published_strategy_sets_and_labels` — Verifies exact compiled sizes (`4, 9, 12, 17, 24`) and strategy labels (`C-N, C-P, D-N, L-N`, etc.).
2. `test_methods_equation1_strategy_cardinality` — Verifies closed-form cardinality Eq. (1) across all $4 \times 128 = 512$ mechanism-profile pairs.
3. `test_distinct_games_and_threshold_grammar` — Confirms 70 distinct games per family in `base` and `cnt` grammars, $|S| \in [2, 135]$ in `cnt`, and 45 strategies at RN24 (`cnt`).
4. `test_digest_implies_identical_payoff_tensors` — Confirms exact `0.0` payoff tensor discrepancy across all profiles sharing a canonical SHA-1 digest (280 distinct games).
5. `test_directed_boundary_cases` — Verifies all 8 hand-calculated boundary payoff cases (`forfeit` vs. `void` ostracism, lone survivor, count-contingent threshold $\theta=2$, reward reciprocation).
6. `test_vectorised_vs_member_by_member_payoffs` — Confirms `< 1e-13` agreement between vectorised `pair_payoffs` and reference `group_payoffs` across random parameters, both grammars, and all conventions.
7. `test_mutation_kernels_column_stochastic` — Verifies zero diagonal, non-negativity, and unit column sums for `uniform`, `action`, `trait`, and `redraw` kernels.
8. `test_neutral_limit_exact_benchmark` — Confirms $\rho = 1/M$ and exact cooperation $1/2$ (H4) and $1/3$ (RN24), difference $-1/6$, at $w = 0$.
9. `test_tiny_probability_analytical_benchmark` — Verifies exact agreement with $x_C = (1 + e^{42w})^{-1}$ down to $w = 10$ ($3.95 \times 10^{-183}$).
10. `test_gth_solver_matches_60digit_mpmath` — Confirms `< 2e-16` discrepancy between `np.longdouble` GTH elimination and 60-digit arbitrary-precision `mpmath`.
11. `test_strong_selection_finite_M_gt9` — Verifies exact finite-$M$ strong-selection formula $x_{\mathrm{C\text{-}NP}} = (M+2)/(M+8)$ within `1e-13` at $w = 5,000$.
12. `test_shapley_axioms_and_dividend_recovery` — Verifies efficiency, null-player, symmetry, linearity, interaction symmetry, and exact Shapley recovery from Harsanyi dividends $\phi_i = \sum_{S \ni i} d(S)/|S|$.
13. `test_precedence_shapley_efficiency` — Verifies permission-structure efficiency and gatekeeper credit allocation (`OPT` $\prec$ `LTG, LAG`).
14. `test_mc_shapley_convergence` — Verifies Monte Carlo permutation estimator uncertainty calibration.
15. `test_abm_deterministic_seed_reproducibility` — Confirms bit-for-bit identical trajectories across repeated calls with fixed seed.
16. `test_canonical_data_files_present_and_within_github_size_limits` — Confirms all 21 canonical data files exist and every file is `< 50 MB`.
17. `test_no_hardcoded_local_paths_or_cross_directory_leaks_in_code` — Confirms zero references to original submission folders or personal user paths.

---

## 4. Item-by-Item Comparison of Manuscript Reported Values vs. Recomputed Values

Every quantitative claim in the Abstract, Main Text, Table 1, and Supplementary Tables 1–13 was recomputed from node-level data (and independently checked live at reference parameters):

| Location | Quantity | Manuscript Reported | Recomputed (Full Precision) | Rounded Match |
| :--- | :--- | :--- | :--- | :--- |
| Abstract / Results | Distinct compiled games per mechanism family | `70` | `70` (`punish`, `reward`, `exclude`, `pool`) | Exact |
| Results / SI Note 1 | Compiled strategy count range (`base` / `cnt`) | `2–48` / `2–135` | `[2, 48]` / `[2, 135]` (`45` at RN24 `cnt`) | Exact |
| Results / SI Table 1 | Payoff kernel max error (9,600 random / 615,870 exhaustive) | `3e-14` / `3.6e-15` | `2.842e-14` / `3.553e-15` | Exact |
| Results / SI Table 1d | Stationary GTH solver vs. 60-digit `mpmath` max diff | `< 2e-16` (`1.1e-16`) | `1.110e-16` | Exact |
| Results / Fig. 1c | GT9 strong-selection error ($M = 10^2, 10^3, 10^4$, $w=1000$) | `5.6e-2`, `6.0e-3`, `7.1e-4` | `5.556e-02`, `5.952e-03`, `7.097e-04` | Exact |
| Results / Fig. 2a | Maximum peer-punishment cooperation / compulsory max (`OPT=0`) | `0.916` / `< 2e-10` | `0.916050` (node `1100010`) / `1.854e-10` | Exact |
| Abstract / Table 1 | Stationary cooperation at 4, 9, 12, 17, 24 strategy sets | `0.913, 0.913, 0.627, 0.466, 0.321` | `0.912687, 0.912687, 0.627051, 0.466444, 0.321222` | Exact |
| Table 1 / Results | Stationary antisocial punishers at 24-strategy node (RN24) | `0.410` (`0.41`) | `0.409644` | Exact |
| Table 1 / Results | Relaxation time (H4 $\to$ RN24, mutant arrivals) | `33` $\to$ `913` (`89, 577, 603`) | `33.2, 88.8, 577.4, 603.4, 913.0` | Exact |
| Results / Fig. 3a | Cooperation drop H4 $\to$ RN24 | `0.591` | `0.591465` | Exact |
| Abstract / Fig. 3a | Uniform dispute $\phi$: `LTG`, `DAG`, `LAG`, `ANTI` (% of drop) | `-0.248 (42%), -0.160 (27%), -0.113 (19%), -0.070 (12%)` | `-0.248169 (41.96%), -0.159712 (27.00%), -0.113407 (19.17%), -0.070177 (11.86%)` | Exact |
| Results | Dispute $\phi_{\text{anti}}(\text{ANTI})$ for antisocial punishers | `+0.204` | `+0.204127` | Exact |
| Results / Fig. 3c | Pairwise interactions $I(\text{DAG},\text{LTG})$ & $I(\text{ANTI},\text{LAG})$ | `-0.38` & `-0.26` | `-0.380289` & `-0.255754` | Exact |
| Results / SI Table 3 | Harsanyi dividends $d(\{\text{DAG},\text{LTG}\})$, $d(\{\text{ANTI},\text{LAG}\})$, 3- & 4-way | `-0.52`, `-0.29`, `+0.14`, `+0.23` | `-0.524088`, `-0.289262`, `+0.136976`, `+0.228661` | Exact |
| Results / SI Table 6 | LTG most negative: Design A (`83%`), Design B (`59%`; $w\ge 1$: `91%`, $0.1\le w<1$: `70%`, $w<0.1$: `13%`) | `83%` / `59%` (`91%, 70%, 13%`) | `83.25%` / `58.55%` (`91.03%, 70.13%, 12.65%`) | Exact |
| Results / Fig. 4c | Reference $w$-crossing interval for $\phi(\text{LTG})$ (uniform) | `[0.032, 0.042]` | `[0.031623, 0.042170]` | Exact |
| Results / SI Table 6 | Sets with $\phi(\text{LTG}) > 0$ at $w \ge 0.1$: Design A / Design B | `70 of 2,000` / `46 of 1,352` | `70 of 2,000` / `46 of 1,352` | Exact |
| Results / SI Table 2 | Neutral ($w=0$, uniform) $\phi$: `LTG`, `ANTI`, `DAG`, `LAG` | `+0.065, +0.065, -0.148, -0.148` | `+0.064815, +0.064815, -0.148148, -0.148148` | Exact |
| Results / Fig. 4a | Action-balanced $\phi$: `LTG, DAG, LAG, ANTI` | `-0.245, -0.150, -0.150, -0.069` | `-0.245424, -0.149964, -0.149964, -0.069092` | Exact |
| Results / Fig. 4a | Single-trait (reset): $v_4$, $\phi(\text{DAG})$, $\phi(\text{LTG})$ | `0.389`, `-0.127`, `+0.042` | `0.389451`, `-0.126877`, `+0.041984` | Exact |
| Results / Fig. 4a | Single-trait (redraw) $\phi$: `LAG, LTG, ANTI, DAG` | `-0.201, -0.167, -0.154, -0.082` | `-0.201189, -0.166558, -0.153864, -0.081523` | Exact |
| Results / Fig. 5a | Ostracism conditional $\phi(\text{ANTI})$ / forfeit vs. void dispute $\phi(\text{ANTI})$ | `-0.653` / `-0.862` vs. `-0.657` | `-0.653411` / `-0.861742` vs. `-0.657104` | Exact |
| Results / SI Table 2 | Pool-fee convention dispute $\phi(\text{ANTI})$: `always` vs. `ifgame` | `-0.124` vs. `-0.217` | `-0.123523` vs. `-0.216560` | Exact |
| Results / SI Table 5 | Global 128-node $\phi(\text{OPT}), \phi(\text{OBS})$ / precedence $\phi_{\text{prec}}(\text{OBS}), \phi_{\text{prec}}(\text{LTG})$ | `+0.377, +0.228` / `+0.383, -0.176` | `+0.377051, +0.228290` / `+0.383287, -0.175722` | Exact |
| Results / Fig. 5b | Extended $2^9$ subcube $\phi$: `LTG, DAG, LAG, ANTI` vs. `MHI, wHI, rHI, bHI` | `-0.202, -0.166, -0.141, -0.131` vs. `+0.034, +0.017, +0.010, +0.0004` | `-0.2024, -0.1664, -0.1411, -0.1315` vs. `+0.0343, +0.0175, +0.0098, +0.0004` | Exact |
| Results / ED Fig. 3 | MC Shapley RMSE ($10^3 / 10^4$) & 95% CI coverage ($10 / 10^3 / 10^4$) | `0.0045 / 0.0014` & `80% / 95.9% / 94.1%` | `0.004519 / 0.001417` & `80.45% / 95.91% / 94.09%` | Exact |
| Results / ED Fig. 5 | Cross-society Kendall $\tau_b$ ($P_{\text{Holm}}$): P1, P4, P2, P3, P5 | `-0.48 (0.015), -0.58 (0.0022), -0.31 (0.057), 0.36 (0.057), -0.09` | `-0.476996 (0.0150), -0.583333 (0.0022), -0.309629 (0.0569), +0.359839 (0.0569), -0.092052` | Exact |

---

## 5. Figure & Working-Directory Isolation Verification

1. **Pixel-Level & Visual Figure Verification (`Fig1`–`Fig5` & `ED_Fig1`–`ED_Fig7`)**:
   All 12 figures were regenerated in `outputs/figures/` (both `.pdf` and 300-dpi `.png`), visually inspected for layout/typography/panel-label integrity, and compared pixel-by-pixel against `01_manuscript/figures/*.png`. Every single one of the 12 PNG figures matched the manuscript reference PNGs with **identical pixel dimensions** and **`0.00000` normalised mean absolute pixel difference**.
2. **Working-Directory (`cwd`) Independence & Filesystem Isolation**:
   Scripts were invoked from outside the repository (`C:\Windows\Temp`) and under a Python `sys.addaudithook` monitor forbidding any read/stat/open access to `submission_NCS_affordance_calculus_rev1`. Zero `FileNotFoundError` or cross-directory accesses occurred.

---

## 6. Pre-Submission Technical Notes for the Authors

No discrepancy was found between the scientific code outputs and any figure, table, or number reported in `manuscript.tex` or `si.tex`. Three minor technical items are noted for the authors' awareness prior to final upload:

1. **Confirm Open-Source Software License**: Add an OSI-approved `LICENSE` file (e.g., `MIT` or `BSD-3-Clause`) to the root of `affordance-calculus-reproducibility/` and update Section 11 of `README.md` before public GitHub / Zenodo deposition, as required by *Nature Computational Science*.
2. **Replace Repository DOI & Author Placeholders**: Once archived on Zenodo, replace `[Repository DOI to be added on acceptance.]` in the manuscript's *Data availability* and *Code availability* statements and update the BibTeX entry in `README.md`.
3. **Minor Cache Artefact in Original `paper_numbers.json` (Unquoted Keys Only)**: In the original `06_code_and_data/code/make_numbers.py`, tie-rounding `.round(10)` for `tau_anti_gain_optobs` was patched via `python code/make_numbers.py extra` only for `punish:uniform` (the sole value quoted in the manuscript, `-0.37`, which matches identically), leaving five *unquoted* `family:kernel` keys in the original cached `paper_numbers.json` from an earlier unrounded run. Running the full `reproduce_results.py --quick` pipeline applies `.round(10)` consistently across all `family:kernel` combinations; no manuscript or SI number is affected.
