"""Unit tests for evolutionary dynamics, fixation probabilities, mutation
kernels, and analytical limits."""
from __future__ import annotations

import sys
from pathlib import Path

import mpmath as mp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import affordance as af


def test_mutation_kernels_column_stochastic():
    alpha = dict(OPT=1, OBS=1, ANTI=1, LTG=1, DAG=1, LAG=1, RET=1)
    for gram in ["base", "cnt"]:
        ss = af.compile_node("punish", alpha, gram)
        for ker in af.ALL_KERNELS:
            K = af.mutation_kernel(ss, ker)
            assert np.allclose(np.diag(K), 0.0)
            assert np.allclose(K.sum(axis=0), 1.0, atol=1e-14)
            assert np.all(K >= 0.0)


def test_neutral_limit_exact_benchmark():
    """At w=0, fixation probabilities equal 1/M and uniform mutation gives
    cooperation 1/2 at 4-strategy node and 1/3 at 24-strategy node (diff = -1/6)."""
    _, rho4, pi4, out4 = af.solve("punish", dict(OPT=1, OBS=1), dict(w=0.0), kernel="uniform")
    _, rho24, pi24, out24 = af.solve("punish", dict(OPT=1, OBS=1, ANTI=1, LTG=1, DAG=1, LAG=1), dict(w=0.0), kernel="uniform")
    M = af.DEFAULTS["M"]
    off4 = rho4[~np.eye(len(pi4), dtype=bool)]
    assert np.allclose(off4, 1.0 / M, atol=1e-14)
    assert abs(out4["coop"] - 0.5) < 1e-14
    assert abs(out24["coop"] - (1.0 / 3.0)) < 1e-14
    assert abs((out24["coop"] - out4["coop"]) + (1.0 / 6.0)) < 1e-14


def test_tiny_probability_analytical_benchmark():
    for w in [0.1, 1.0, 5.0, 10.0]:
        _, _, _, out = af.solve("punish", {}, dict(w=w))
        exact = float(np.exp(-np.logaddexp(0.0, 42.0 * w)))
        rel_err = abs(out["coop"] - exact) / max(exact, 1e-300)
        assert rel_err < 1e-12


def test_gth_solver_matches_60digit_mpmath():
    mp.mp.dps = 60
    ss = af.compile_node("punish", dict(OPT=1, OBS=1, ANTI=1, LTG=1, DAG=1, LAG=1))
    par = {**af.DEFAULTS, "M": 100, "w": 1.0}
    L = af.log_fixation_matrix(ss, par)
    S = L.shape[0]
    kap = af.mutation_kernel(ss, "uniform")
    P = [[mp.mpf(0) if i == j else mp.mpf(kap[j, i]) * mp.e ** mp.mpf(L[j, i]) for j in range(S)] for i in range(S)]
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
    tot = mp.fsum(pim)
    pi_exact = np.array([float(x / tot) for x in pim])
    pi_gth = af.stationary(af.fixation_matrix(ss, par), kap)
    assert np.max(np.abs(pi_exact - pi_gth)) < 2e-16


def test_strong_selection_finite_M_gt9():
    alpha = dict(OPT=1, OBS=1, ANTI=1, DAG=1)
    for M in [20, 50, 100]:
        ss, _, pi, _ = af.solve("punish", alpha, dict(M=M, w=5000.0))
        abund = dict(zip(ss.labels, pi))
        target = (M + 2) / (M + 8)
        assert abs(abund["C-NP"] - target) < 1e-13
