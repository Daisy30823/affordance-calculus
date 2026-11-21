"""Finite-mutation individual-based check of the rare-mutation results.

Moran birth-death process in a well-mixed population of M individuals. At each
step one individual reproduces with probability proportional to exp(w * pi),
where pi is its expected payoff given the current composition (the focal
individual is excluded from its own co-player sample, groups of n drawn without
replacement), the offspring adopts a uniformly chosen other compiled strategy
with probability mu, and it replaces a uniformly chosen individual. Expected
payoffs are computed exactly from a pre-tabulated payoff for every focal
strategy and co-player multiset.
"""
from __future__ import annotations

import itertools
import time
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

from . import RESULTS_OUT_DIR, affordance as af

MUS = [0.001, 0.005, 0.02]
SEEDS = [1, 2]
T_STEPS = 10_000_000
BURN = 500_000


def payoff_table(alpha, par):
    p = {**af.DEFAULTS, **par}
    n = int(p["n"])
    ss = af.compile_node("punish", alpha)
    S = len(ss.strategies)
    m = n - 1
    nrank = comb(S + m - 1, m)
    tab = np.zeros((S, nrank))
    for ms in itertools.combinations_with_replacement(range(S), m):
        rk = sum(comb(t + j, j + 1) for j, t in enumerate(ms))
        for x in range(S):
            tab[x, rk] = af.group_payoffs(ss, [x] + list(ms), p)[0]
    isC = np.array([s.action == "C" for s in ss.strategies])
    return tab, isC


@njit
def _lcomb(a, b, lf):
    if b < 0 or b > a:
        return -1e300
    return lf[a] - lf[b] - lf[a - b]


@njit
def simulate(tab, isC, M, n, w, mu, T, burn, seed):
    np.random.seed(seed)
    S = tab.shape[0]
    lf = np.zeros(M + 1)
    for i in range(1, M + 1):
        lf[i] = lf[i - 1] + np.log(i)
    lt = lf[M - 1] - lf[n - 1] - lf[M - n]
    counts = np.zeros(S, np.int64)
    counts[np.random.randint(S)] = M
    present = np.zeros(S, np.int64)
    fit = np.zeros(S)
    ms = np.zeros(4, np.int64)
    occ = 0.0
    nC = 0
    for t in range(S):
        if isC[t]:
            nC += counts[t]
    changed = True
    npres = 0
    for step in range(T):
        if changed:
            npres = 0
            for t in range(S):
                if counts[t] > 0:
                    present[npres] = t
                    npres += 1
            tot_f = 0.0
            for ix in range(npres):
                x = present[ix]
                counts[x] -= 1
                ep = 0.0
                # enumerate multisets of 4 co-players over present types (n = 5)
                for a in range(npres):
                    for b in range(a, npres):
                        for c in range(b, npres):
                            for d in range(c, npres):
                                ms[0] = present[a]; ms[1] = present[b]; ms[2] = present[c]; ms[3] = present[d]
                                lp = -lt
                                # multiplicities
                                j = 0
                                while j < 4:
                                    k = j
                                    while k + 1 < 4 and ms[k + 1] == ms[j]:
                                        k += 1
                                    lp += _lcomb(counts[ms[j]], k - j + 1, lf)
                                    j = k + 1
                                if lp < -700:
                                    continue
                                rk = 0
                                for jj in range(4):
                                    tj = ms[jj] + jj
                                    # comb(tj, jj+1)
                                    v = 1.0
                                    for q in range(jj + 1):
                                        v = v * (tj - q) / (q + 1)
                                    rk += int(round(v))
                                ep += np.exp(lp) * tab[x, rk]
                counts[x] += 1
                fit[ix] = counts[x] * np.exp(w * ep)
                tot_f += fit[ix]
            for ix in range(1, npres):
                fit[ix] += fit[ix - 1]
            for ix in range(npres):
                fit[ix] /= tot_f
        u = np.random.random()
        ix = 0
        while ix < npres - 1 and fit[ix] < u:
            ix += 1
        parent = present[ix]
        child = parent
        if np.random.random() < mu:
            child = np.random.randint(S - 1)
            if child >= parent:
                child += 1
        u2 = np.random.randint(M)
        cum = 0
        dead = present[0]
        for ix in range(npres):
            cum += counts[present[ix]]
            if u2 < cum:
                dead = present[ix]
                break
        changed = dead != child
        if changed:
            counts[dead] -= 1
            counts[child] += 1
            if isC[child]:
                nC += 1
            if isC[dead]:
                nC -= 1
        if step >= burn:
            occ += nC / M
    return occ / (T - burn)


def jobs(nodes=None, mus=None, seeds=None):
    mus = MUS if mus is None else mus
    seeds = SEEDS if seeds is None else seeds
    if nodes is None:
        node_keys = [af.node_key(alpha) for alpha in af.all_nodes(fixed=dict(OPT=1, OBS=1, RET=0))]
    else:
        node_keys = list(nodes)
    out = []
    for key in node_keys:
        for mu in mus:
            for seed in seeds:
                out.append((key, mu, seed))
    return out


def run_abm_shard(i: int, k: int, shards_dir: Path | None = None,
                  t_steps: int = T_STEPS, burn: int = BURN,
                  nodes=None, mus=None, seeds=None) -> pd.DataFrame:
    sh = Path(shards_dir) if shards_dir is not None else (RESULTS_OUT_DIR / "shards")
    sh.mkdir(parents=True, exist_ok=True)
    par = dict(af.DEFAULTS)
    rows = []
    tabs = {}
    for key, mu, seed in jobs(nodes=nodes, mus=mus, seeds=seeds)[i::k]:
        alpha = dict(zip(af.PRIMS, map(int, key)))
        if key not in tabs:
            tabs[key] = payoff_table(alpha, par)
        tab, isC = tabs[key]
        t = time.time()
        sim_seed = 1000 * seed + int(mu * 1e4)
        c = simulate(tab, isC, int(par["M"]), int(par["n"]), float(par["w"]), mu, t_steps, burn, sim_seed)
        rows.append(dict(node=key, mu=mu, seed=seed, coop=float(c), T=t_steps, burn=burn, sec=time.time() - t))
        pd.DataFrame(rows).to_csv(sh / f"abm_{i}.csv", index=False, quoting=1)
    return pd.DataFrame(rows)
