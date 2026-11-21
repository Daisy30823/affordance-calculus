"""Lattice evaluations, validation checks, parameter sweeps, selection scans,
extended lattices, and convention sensitivity checks.

Consolidates the model-execution routines of the manuscript while preserving
all mathematical definitions, random seeds, parameter ranges, and numerical
tolerances identically.
"""
from __future__ import annotations

import io
import itertools
import json
import time
from pathlib import Path

import mpmath as mp
import numpy as np
import pandas as pd

from . import RESULTS_OUT_DIR, abm, affordance as af

NAMED = {
    "H4": dict(OPT=1, OBS=1),
    "GT9": dict(OPT=1, OBS=1, ANTI=1, DAG=1),
    "GT12": dict(OPT=1, OBS=1, ANTI=1, DAG=1, LAG=1),
    "GT17": dict(OPT=1, OBS=1, ANTI=1, DAG=1, LTG=1),
    "RN24": dict(OPT=1, OBS=1, ANTI=1, DAG=1, LTG=1, LAG=1),
}
N_SWEEP = 2000
SWEEP_SEEDS = {"A": 20261109, "B": 20261110}
SUB = dict(OPT=1, OBS=1)  # operative subcube (exit and observability on)
EXT9 = ["ANTI", "LTG", "DAG", "LAG", "RET", "rHI", "bHI", "wHI", "MHI"]
EXT11 = af.PRIMS + ["rHI", "bHI", "wHI", "MHI"]


def full(a: dict) -> dict:
    return {k: int(a.get(k, 0)) for k in af.PRIMS}


def lattice_rows(par, fams=af.FAMILIES, kernels=af.KERNELS, grammar="base",
                 fixed=None, extra=False, tag=None):
    rows = []
    for fam in fams:
        for alpha in af.all_nodes(fixed=fixed):
            p = {**af.DEFAULTS, **par}
            ss = af.compile_node(fam, alpha, grammar)
            rho = af.fixation_matrix(ss, p)
            for ker in kernels:
                kap = af.mutation_kernel(ss, ker)
                pi = af.stationary(rho, kap)
                out = af.outcomes(ss, pi, p)
                if extra:
                    out["relax"] = af.relaxation_time(rho, kap)
                rows.append(dict(tag=tag, family=fam, grammar=grammar, excl=p["excl"], kernel=ker,
                                 w=p["w"], n=p["n"], M=p["M"], node=af.node_key(alpha), **alpha,
                                 nstrat=len(ss.strategies), digest=ss.digest(), **out))
    return rows


# ---------------------------------------------------------------------------- validation & core
def directed_tests():
    """Hand-computed payoffs for boundary cases (reference parameters)."""
    p = dict(af.DEFAULTS)
    res = []
    # ostracism, all participants excluded: five C players targeting C
    ss = af.compile_node("exclude", dict(OPT=1, OBS=1, ANTI=1))
    s = [i for i, x in enumerate(ss.strategies) if x.action == "C" and x.targets == frozenset({"C"})][0]
    for conv, expect in [("forfeit", -1.0 - 0.3 * 4), ("void", 3.0 - 1.0 - 0.3 * 4)]:
        got = af.group_payoffs(ss, [s] * 5, {**p, "excl": conv})[0]
        res.append(dict(case=f"exclude all-excluded ({conv})", expected=expect, got=float(got)))
    # ostracism, one survivor: four C targeting C + one D-N survivor
    sD = [i for i, x in enumerate(ss.strategies) if x.action == "D" and not x.targets][0]
    pay = af.group_payoffs(ss, [s] * 4 + [sD], p)
    res.append(dict(case="exclude single non-excluded defector", expected=3.0 * 4 / 1, got=float(pay[-1])))
    # pooled class targetable only if both C and D permitted (OBS=0)
    ssP = af.compile_node("punish", dict(OPT=1, OBS=0, ANTI=0, DAG=1))
    res.append(dict(case="OBS=0, ANTI=0: no permitted class -> no agency", expected=3, got=len(ssP.strategies)))
    ssP = af.compile_node("punish", dict(OPT=1, OBS=0, ANTI=1, DAG=1))
    res.append(dict(case="OBS=0, ANTI=1: pooled class targetable", expected=5, got=len(ssP.strategies)))
    # count-contingent punishment: C-P* punishes a lone defector? (needs >=2 D co-players)
    ssC = af.compile_node("punish", dict(OPT=1, OBS=1), "cnt")
    cp2 = [i for i, x in enumerate(ssC.strategies) if x.action == "C" and x.targets and x.theta == 2][0]
    dn = [i for i, x in enumerate(ssC.strategies) if x.action == "D"][0]
    pay1 = af.group_payoffs(ssC, [cp2] * 4 + [dn], p)
    pay2 = af.group_payoffs(ssC, [cp2] * 3 + [dn] * 2, p)
    res.append(dict(case="threshold punisher ignores lone defector (D payoff)", expected=3 * 4 / 5, got=float(pay1[-1])))
    res.append(dict(case="threshold punisher fines two defectors (each D payoff)", expected=3 * 3 / 5 - 3 * 1.0, got=float(pay2[-1])))
    # reciprocated reward
    ssR = af.compile_node("reward", dict(OPT=1, OBS=1, RET=1))
    cr = [i for i, x in enumerate(ssR.strategies) if x.action == "C" and x.targets and not x.ret][0]
    cq = [i for i, x in enumerate(ssR.strategies) if x.action == "C" and x.targets and x.ret][0]
    pay = af.group_payoffs(ssR, [cr, cq, cq, cq, cq], p)
    res.append(dict(case="reward reciprocation", expected=2 - 1.2 + 4 + 4, got=float(pay[0])))
    for r_ in res:
        r_["pass"] = bool(abs(r_["expected"] - r_["got"]) < 1e-12)
    return res


def run_validation(out_dir: Path | None = None, max_M_limit: int = 10000) -> dict:
    """Run analytical limit, finite-M, tiny-probability, 60-digit mpmath, and kernel checks."""
    val = {"sizes": {k: len(af.compile_node("punish", full(a)).strategies) for k, a in NAMED.items()}}
    val["directed_tests"] = directed_tests()
    rng = np.random.default_rng(7); me = 0.0; nt = 0
    for fam in af.FAMILIES:
        for gram in ["base", "cnt"]:
            for _ in range(60):
                alpha = {k: int(rng.integers(2)) for k in af.PRIMS}
                ss = af.compile_node(fam, alpha, gram); S = len(ss.strategies)
                par = {**af.DEFAULTS, **dict(r=rng.uniform(1.5, 5), gamma=rng.uniform(.05, 1), beta=rng.uniform(.2, 3),
                                             G=rng.uniform(.05, 1), rho_R=rng.uniform(.2, 3), sigma=rng.uniform(0, 1),
                                             excl=str(rng.choice(["forfeit", "void"])), n=int(rng.integers(3, 9)))}
                PX, PY = af.pair_payoffs(ss, par); n = par["n"]
                for _ in range(20):
                    x, y = rng.integers(S, size=2); a_ = int(rng.integers(1, n))
                    ref = af.group_payoffs(ss, [x] * a_ + [y] * (n - a_), par)
                    me = max(me, abs(ref[0] - PX[x, y, a_ - 1]))
                    if x != y:
                        me = max(me, abs(ref[-1] - PY[x, y, a_ - 1]))
                    nt += 1
    val["kernel_test"] = dict(n_groups=nt, max_abs_err=float(me))

    analytic = {
        "GT9": {"C-NP": 1.0},
        "GT12": {"C-NP": 2 / 3, "L-PN": 1 / 4, "L-PP": 1 / 12},
        "GT17": {"D-NNP": 3 / 8, "C-NPN": 3 / 8, "D-PNP": 1 / 8, "C-NPP": 1 / 8},
        "RN24": {"D-NNP": 1 / 4, "C-NPN": 1 / 4, "L-PNN": 1 / 4, "D-PNP": 1 / 12, "C-NPP": 1 / 12, "L-PPN": 1 / 12},
    }
    Ms = [m for m in [100, 1000, 10000] if m <= max_M_limit]
    vt = []
    for nm, target in analytic.items():
        for M in Ms:
            for w in [10.0, 30.0, 100.0, 300.0, 1000.0]:
                ss, rho, pi, out = af.solve("punish", full(NAMED[nm]), dict(w=w, M=M))
                lab = dict(zip(ss.labels, pi))
                tv = sum(abs(lab.get(k, 0) - target.get(k, 0)) for k in set(lab) | set(target)) / 2
                vt.append(dict(node=nm, M=M, w=w, max_abs_err=float(max(abs(lab[k] - v) for k, v in target.items())), tv=float(tv)))
    val["limits"] = vt

    fm = []
    for w in [200.0, 1000.0, 5000.0]:
        for M in [20, 50, 100, 200, 500]:
            ss, rho, pi, out = af.solve("punish", full(NAMED["GT9"]), dict(w=w, M=M))
            lab = dict(zip(ss.labels, pi))
            fm.append(dict(w=w, M=M, x=float(lab["C-NP"]), target=float((M + 2) / (M + 8)),
                           err=float(abs(lab["C-NP"] - (M + 2) / (M + 8)))))
    val["finiteM"] = fm

    bm = []
    for w in [0.1, 1.0, 5.0, 10.0]:
        ss, rho, pi, out = af.solve("punish", full({}), dict(w=w))
        bm.append(dict(w=w, coop=float(out["coop"]), exact=float(np.exp(-np.logaddexp(0, 42 * w)))))
    val["tiny_benchmark"] = bm

    mp.mp.dps = 60
    hp = []
    for nm, M, w in [("RN24", 1000, 100.0), ("GT17", 1000, 100.0), ("GT9", 100, 1.0), ("RN24", 100, 1.0)]:
        ss = af.compile_node("punish", full(NAMED[nm])); p = {**af.DEFAULTS, "w": w, "M": M}
        L = af.log_fixation_matrix(ss, p); S = L.shape[0]
        kap = af.mutation_kernel(ss, "uniform")
        Q = [[mp.mpf(0) if i == j else mp.mpf(kap[j, i]) * mp.e ** mp.mpf(L[j, i]) for j in range(S)] for i in range(S)]
        P = [row[:] for row in Q]
        for k in range(S - 1, 0, -1):
            s = mp.fsum(P[k][:k])
            for i in range(k):
                P[i][k] /= s
            for i in range(k):
                for j in range(k):
                    P[i][j] += P[i][k] * P[k][j]
        pim = [mp.mpf(1)] + [mp.mpf(0)] * (S - 1)
        for k in range(1, S):
            pim[k] = mp.fsum(pim[i] * P[i][k] for i in range(k))
        tot = mp.fsum(pim); pim = [float(x / tot) for x in pim]
        pid = af.stationary(af.fixation_matrix(ss, p), kap)
        hp.append(dict(node=nm, M=M, w=w, max_abs_diff=float(np.max(np.abs(np.array(pim) - pid)))))
    val["high_precision"] = hp

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "validation.json", "w", encoding="utf-8") as fh:
            json.dump(val, fh, indent=1, default=float)
    return val


def run_core(out_dir: Path | None = None, max_M_limit: int = 10000, verbose: bool = True) -> None:
    out = Path(out_dir) if out_dir is not None else RESULTS_OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    run_validation(out, max_M_limit=max_M_limit)
    if verbose:
        print(f"  validation completed ({time.time() - t0:.1f}s)", flush=True)

    rows = []
    rows += lattice_rows({}, extra=True, tag="ref")
    rows += lattice_rows({"w": 0.0}, tag="neutral")
    rows += lattice_rows({}, fams=["punish"], grammar="cnt", tag="ref_cnt")
    rows += lattice_rows({"excl": "void"}, fams=["exclude"], tag="ref_void")
    rows += lattice_rows({"n": 4}, fams=["punish"], tag="ref_n4")
    pd.DataFrame(rows).to_csv(out / "lattice_reference.csv", index=False)
    if verbose:
        print(f"  lattice_reference.csv written ({time.time() - t0:.1f}s)", flush=True)

    rows = []
    for w in np.logspace(-3, 2, 41):
        rows += lattice_rows({"w": float(w)}, fams=["punish"], fixed=SUB, tag="wscan")
    pd.DataFrame(rows).to_csv(out / "selection_scan_nodes.csv", index=False)
    if verbose:
        print(f"  selection_scan_nodes.csv written ({time.time() - t0:.1f}s)", flush=True)


def run_core_redraw(out_dir: Path | None = None, verbose: bool = True) -> None:
    out = Path(out_dir) if out_dir is not None else RESULTS_OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = []
    rows += lattice_rows({}, kernels=["redraw"], extra=True, tag="ref")
    rows += lattice_rows({"w": 0.0}, kernels=["redraw"], tag="neutral")
    rows += lattice_rows({}, fams=["punish"], kernels=["redraw"], grammar="cnt", tag="ref_cnt")
    rows += lattice_rows({"excl": "void"}, fams=["exclude"], kernels=["redraw"], tag="ref_void")
    rows += lattice_rows({"n": 4}, fams=["punish"], kernels=["redraw"], tag="ref_n4")
    pd.DataFrame(rows).to_csv(out / "lattice_reference_redraw.csv", index=False)

    rows = []
    for w in np.logspace(-3, 2, 41):
        rows += lattice_rows({"w": float(w)}, fams=["punish"], kernels=["redraw"], fixed=SUB, tag="wscan")
    pd.DataFrame(rows).to_csv(out / "selection_scan_nodes_redraw.csv", index=False)
    if verbose:
        print(f"  redraw lattices & w-scan written ({time.time() - t0:.1f}s)", flush=True)


# ---------------------------------------------------------------------------- sweeps
def draw(design: str, rng: np.random.Generator) -> dict:
    if design == "A":       # the original box
        n, M = 5, 100
        r = rng.uniform(1.5, 5.0); lw = rng.uniform(-1, 1)
    else:                   # expanded box
        n = int(rng.integers(3, 9)); M = int(rng.choice([50, 100, 200]))
        r = rng.uniform(1.5, n); lw = rng.uniform(-2, 1)
    g = rng.uniform(0.05, 1.0); b = rng.uniform(0.2, 3.0)
    return dict(n=n, M=M, r=r, sigma=rng.uniform(0.0, r - 1.0), gamma=g, beta=b, G=g, rho_R=b, w=float(10 ** lw))


def sweep_shard(design: str, i: int, k: int, shards_dir: Path | None = None,
                n_draws: int = N_SWEEP) -> Path:
    sh = Path(shards_dir) if shards_dir is not None else (RESULTS_OUT_DIR / "shards")
    sh.mkdir(parents=True, exist_ok=True)
    # Always generate the full 2,000-element seed state so draw j matches full sweep index j
    seeds = np.random.SeedSequence(SWEEP_SEEDS[design]).generate_state(max(N_SWEEP, n_draws)).tolist()
    pars, recs = [], []
    for j in range(i, n_draws, k):
        rng = np.random.default_rng(seeds[j]); par = draw(design, rng)
        pars.append(dict(idx=j, **par))
        for fam in af.FAMILIES:
            for alpha in af.all_nodes():
                ss = af.compile_node(fam, alpha); p = {**af.DEFAULTS, **par}
                rho = af.fixation_matrix(ss, p)
                for ki, ker in enumerate(af.KERNELS):
                    pi = af.stationary(rho, af.mutation_kernel(ss, ker))
                    o = af.outcomes(ss, pi)
                    recs.append((j, af.FAMILIES.index(fam), 0, ki, int(af.node_key(alpha), 2), o["coop"], o["anti"]))
        for fam, var, extra in [("punish", 1, dict(grammar="cnt")), ("exclude", 2, dict(excl="void"))]:
            for alpha in af.all_nodes(fixed=SUB):
                p = {**af.DEFAULTS, **par}
                if "excl" in extra:
                    p["excl"] = extra["excl"]
                ss = af.compile_node(fam, alpha, extra.get("grammar", "base"))
                pi = af.stationary(af.fixation_matrix(ss, p), af.mutation_kernel(ss, "uniform"))
                o = af.outcomes(ss, pi)
                recs.append((j, af.FAMILIES.index(fam), var, 0, int(af.node_key(alpha), 2), o["coop"], o["anti"]))
    out_file = sh / f"sweep{design}_{i}.npz"
    np.savez_compressed(out_file, recs=np.array(recs), pars=pd.DataFrame(pars).to_json())
    return out_file


def sweep_redraw_shard(design: str, i: int, k: int, shards_dir: Path | None = None,
                       n_draws: int = N_SWEEP) -> Path:
    sh = Path(shards_dir) if shards_dir is not None else (RESULTS_OUT_DIR / "shards")
    sh.mkdir(parents=True, exist_ok=True)
    seeds = np.random.SeedSequence(SWEEP_SEEDS[design]).generate_state(max(N_SWEEP, n_draws)).tolist()
    recs = []
    for j in range(i, n_draws, k):
        rng = np.random.default_rng(seeds[j]); par = draw(design, rng)
        for fam in af.FAMILIES:
            for alpha in af.all_nodes(fixed=SUB):
                ss = af.compile_node(fam, alpha); p = {**af.DEFAULTS, **par}
                pi = af.stationary(af.fixation_matrix(ss, p), af.mutation_kernel(ss, "redraw"))
                o = af.outcomes(ss, pi)
                recs.append((j, af.FAMILIES.index(fam), 0, 3, int(af.node_key(alpha), 2), o["coop"], o["anti"]))
    out_file = sh / f"sweepR{design}_{i}.npz"
    np.savez_compressed(out_file, recs=np.array(recs))
    return out_file


# ---------------------------------------------------------------------------- extended lattices
def ext_par(d: dict, r_hi: float = 4.0) -> dict:
    return dict(r=r_hi if d["rHI"] else 3.0, beta=3.0 if d["bHI"] else 1.0,
                w=10.0 if d["wHI"] else 1.0, M=1000 if d["MHI"] else 100)


def ext_jobs(limit: int | None = None):
    jobs = [("11", "".join(map(str, bits))) for bits in itertools.product([0, 1], repeat=len(EXT11))]
    return jobs if limit is None else jobs[:limit]


def ext_shard(i: int, k: int, shards_dir: Path | None = None, r_hi: float = 4.0,
              prefix: str = "ext", limit: int | None = None) -> Path:
    sh = Path(shards_dir) if shards_dir is not None else (RESULTS_OUT_DIR / "shards")
    sh.mkdir(parents=True, exist_ok=True)
    out = []
    for _, key in ext_jobs(limit=limit)[i::k]:
        d = dict(zip(EXT11, map(int, key)))
        alpha = {p: d[p] for p in af.PRIMS}
        ss, rho, pi, o = af.solve("punish", alpha, ext_par(d, r_hi=r_hi))
        out.append(dict(node=key, coop=o["coop"], anti=o["anti"]))
    out_file = sh / f"{prefix}_{i}.csv"
    pd.DataFrame(out).to_csv(out_file, index=False, quoting=1)
    return out_file


def merge_shards(out_dir: Path | None = None, shards_dir: Path | None = None,
                 n_draws: int = N_SWEEP, n_ext: int | None = None,
                 n_abm_jobs: int | None = None) -> None:
    out = Path(out_dir) if out_dir is not None else RESULTS_OUT_DIR
    sh = Path(shards_dir) if shards_dir is not None else (out / "shards")
    out.mkdir(parents=True, exist_ok=True)

    for design in ["A", "B"]:
        fs = sorted(sh.glob(f"sweep{design}_*.npz"))
        if fs:
            recs = [np.load(f, allow_pickle=True) for f in fs]
            R = np.concatenate([r["recs"] for r in recs])
            P = pd.concat([pd.read_json(io.StringIO(str(r["pars"]))) for r in recs]).sort_values("idx")
            assert sorted(P.idx.tolist()) == list(range(n_draws)), design
            per = 4 * 128 * 3 + 32 + 32
            assert len(R) == n_draws * per, (design, len(R))
            np.savez_compressed(out / f"sweep{design}_nodes.npz", recs=R)
            P.to_csv(out / f"sweep{design}_params.csv", index=False)

        fs_r = sorted(sh.glob(f"sweepR{design}_*.npz"))
        if fs_r:
            Rr = np.concatenate([np.load(f)["recs"] for f in fs_r])
            assert len(Rr) == n_draws * 4 * 32 and len(np.unique(Rr[:, 0])) == n_draws, design
            np.savez_compressed(out / f"sweep{design}_nodes_redraw.npz", recs=Rr)

    for prefix, target_name in [("ext", "extended_lattice.csv"),
                                ("ext_r5", "extended_lattice_r5_boundary.csv")]:
        fs_ext = sorted(f for f in sh.glob(f"{prefix}_*.csv") if f.stem[len(prefix) + 1:].isdigit())
        if fs_ext:
            ext_df = pd.concat([pd.read_csv(f, dtype={"node": str}) for f in fs_ext]).sort_values("node")
            expected = (2 ** len(EXT11)) if n_ext is None else n_ext
            assert ext_df.node.nunique() == len(ext_df) == expected, (prefix, len(ext_df), expected)
            ext_df.to_csv(out / target_name, index=False, quoting=1)

    fs_abm = sorted(sh.glob("abm_*.csv"))
    if fs_abm:
        ab_df = pd.concat([pd.read_csv(f, dtype={"node": str}) for f in fs_abm])
        expected_abm = len(abm.jobs()) if n_abm_jobs is None else n_abm_jobs
        assert len(ab_df) == expected_abm, (len(ab_df), expected_abm)
        ab_df.to_csv(out / "abm_finite_mutation.csv", index=False, quoting=1)


# ---------------------------------------------------------------------------- convention & extra checks
def run_pool_convention(out_dir: Path | None = None) -> dict:
    """Sensitivity of the pool-punishment family to the pool-fee convention ('always' vs 'ifgame')."""
    out = Path(out_dir) if out_dir is not None else RESULTS_OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    DISP = ["ANTI", "DAG", "LTG", "LAG"]
    COND = ["ANTI", "LTG", "DAG", "LAG", "RET"]
    ss = af.compile_node("pool", dict(OPT=1, OBS=1))
    sP = [i for i, x in enumerate(ss.strategies) if x.action == "C" and x.targets][0]
    sL = [i for i, x in enumerate(ss.strategies) if x.action == "L"][0]
    tests = {conv: float(af.group_payoffs(ss, [sP] + [sL] * 4, {**af.DEFAULTS, "pool_fee": conv})[0])
             for conv in ["always", "ifgame"]}
    assert abs(tests["always"] - 0.7) < 1e-12 and abs(tests["ifgame"] - 1.0) < 1e-12
    rows = []
    for conv in ["always", "ifgame"]:
        rows += lattice_rows({"pool_fee": conv}, fams=["pool"], kernels=["uniform"], tag=conv)
    L = pd.DataFrame(rows)
    L.to_csv(out / "lattice_pool_convention.csv", index=False)
    res = dict(directed=tests)
    for conv, g in L.groupby("tag"):
        vc = dict(zip(g.node, g.coop))
        sub = {}
        for bits in itertools.product([0, 1], repeat=4):
            d = dict(OPT=1, OBS=1, RET=0); d.update(dict(zip(DISP, bits)))
            sub["".join(map(str, bits))] = vc["".join(str(d.get(k, 0)) for k in af.PRIMS)]
        cs = {}
        for bits in itertools.product([0, 1], repeat=5):
            d = dict(OPT=1, OBS=1); d.update(dict(zip(COND, bits)))
            cs["".join(map(str, bits))] = vc["".join(str(d.get(k, 0)) for k in af.PRIMS)]
        res[conv] = dict(phi=af.exact_shapley(sub, DISP), cond_phi=af.exact_shapley(cs, COND),
                         coop_max=float(g.coop.max()), endpoints=[sub["0000"], sub["1111"]])
    with open(out / "pool_convention.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    return res


def run_extra_checks(out_dir: Path | None = None, grammars=("base", "cnt")) -> dict:
    """Verify (1) digest payoff tensor identity across all 280 distinct games,
    (2) exhaustive two-type group payoff agreement against member-by-member evaluation,
    and (3) top stationary strategies at the 24-strategy node."""
    out = Path(out_dir) if out_dir is not None else RESULTS_OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    worst = 0.0; groups = 0
    for fam in af.FAMILIES:
        by = {}
        for alpha in af.all_nodes():
            ss = af.compile_node(fam, alpha)
            by.setdefault(ss.digest(), []).append(ss)
        for _, lst in by.items():
            groups += 1
            P0 = af.pair_payoffs(lst[0], af.DEFAULTS)
            for ss in lst[1:]:
                P1 = af.pair_payoffs(ss, af.DEFAULTS)
                worst = max(worst, float(np.abs(P0[0] - P1[0]).max()), float(np.abs(P0[1] - P1[1]).max()))
    res["digest_payoff_identity"] = dict(n_games=groups, max_abs_diff=worst)

    worst = 0.0; ncmp = 0
    for grammar in grammars:
        for fam in af.FAMILIES:
            for n in [3, 5]:
                for excl in (["forfeit", "void"] if fam == "exclude" else ["forfeit"]):
                    p = {**af.DEFAULTS, "n": n, "excl": excl}
                    ss = af.compile_node(fam, dict(OPT=1, OBS=1, ANTI=1, LTG=1, DAG=1, LAG=1, RET=1), grammar)
                    PX, PY = af.pair_payoffs(ss, p)
                    S = len(ss.strategies)
                    for x in range(S):
                        for y in range(S):
                            for a in range(1, n):
                                g = af.group_payoffs(ss, [x] * a + [y] * (n - a), p)
                                worst = max(worst, abs(g[0] - PX[x, y, a - 1]), abs(g[-1] - PY[x, y, a - 1]))
                                ncmp += 1
    res["exhaustive_two_type"] = dict(n_comparisons=ncmp, max_abs_err=float(worst))

    ss = af.compile_node("punish", dict(OPT=1, OBS=1, ANTI=1, LTG=1, DAG=1, LAG=1))
    pi = af.stationary(af.fixation_matrix(ss, af.DEFAULTS))
    order = np.argsort(pi)[::-1][:3]
    res["RN24_top"] = [dict(label=ss.labels[i], freq=float(pi[i])) for i in order]
    with open(out / "extra_checks.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    return res
