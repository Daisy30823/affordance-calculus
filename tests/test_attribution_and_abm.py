"""Unit tests for Shapley attribution, interaction indices, Harsanyi dividends,
precedence-constrained values, Monte Carlo Shapley, and ABM determinism."""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import abm, affordance as af


def test_shapley_axioms_and_dividend_recovery():
    players = ["A", "B", "C", "D"]
    rng = np.random.default_rng(123)
    # Construct game where D is a null player and A, B are symmetric
    v = {}
    base_vals = {
        (0, 0, 0): 0.2,
        (1, 0, 0): 0.5,
        (0, 1, 0): 0.5,
        (0, 0, 1): 0.3,
        (1, 1, 0): 0.9,
        (1, 0, 1): 0.65,
        (0, 1, 1): 0.65,
        (1, 1, 1): 0.95,
    }
    for bits in itertools.product([0, 1], repeat=4):
        v["".join(map(str, bits))] = base_vals[bits[:3]]  # D has zero effect

    phi = af.exact_shapley(v, players)
    # Efficiency
    assert abs(sum(phi.values()) - (v["1111"] - v["0000"])) < 1e-15
    # Null player
    assert abs(phi["D"]) < 1e-15
    # Symmetry
    assert abs(phi["A"] - phi["B"]) < 1e-15

    # Linearity
    w = {"".join(map(str, b)): float(rng.uniform(-1, 1)) for b in itertools.product([0, 1], repeat=4)}
    phi_w = af.exact_shapley(w, players)
    comb = {k: 2.5 * v[k] - 1.3 * w[k] for k in v}
    phi_c = af.exact_shapley(comb, players)
    for p in players:
        assert abs(phi_c[p] - (2.5 * phi[p] - 1.3 * phi_w[p])) < 1e-14

    # Harsanyi dividends sum to total change and recover Shapley values
    divs = af.harsanyi_dividends(w, players)
    non_empty_sum = sum(val for k, val in divs.items() if k != "0000")
    assert abs(non_empty_sum - (w["1111"] - w["0000"])) < 1e-14
    for i, p in enumerate(players):
        recovered = sum(val / k.count("1") for k, val in divs.items() if k[i] == "1")
        assert abs(recovered - phi_w[p]) < 1e-14

    # Pairwise interaction symmetry
    inter = af.shapley_interactions(w, players)
    assert np.allclose(inter, inter.T, atol=1e-15)


def test_precedence_shapley_efficiency():
    players = ["OPT", "LTG", "LAG"]
    v = {
        "000": 0.1, "001": 0.1, "010": 0.1, "011": 0.1,  # LTG, LAG inert when OPT=0
        "100": 0.9, "101": 0.7, "110": 0.5, "111": 0.3,
    }
    plain = af.exact_shapley(v, players)
    prec = af.precedence_shapley(v, players, [("OPT", "LTG"), ("OPT", "LAG")])
    assert abs(sum(prec.values()) - (v["111"] - v["000"])) < 1e-15
    # Under precedence, OPT receives full +0.8 initial jump (000 -> 100)
    assert abs(prec["OPT"] - 0.8) < 1e-15
    assert prec["LTG"] < plain["LTG"]


def test_mc_shapley_convergence():
    players = ["A", "B", "C", "D"]
    rng_data = np.random.default_rng(7)
    v = {"".join(map(str, b)): float(rng_data.normal()) for b in itertools.product([0, 1], repeat=4)}
    exact = np.array([af.exact_shapley(v, players)[p] for p in players])
    fn = lambda st: v["".join(str(st[p]) for p in players)]
    mean, se, _ = af.mc_shapley(fn, players, 4000, np.random.default_rng(99))
    assert np.max(np.abs(mean - exact)) < 4.0 * np.max(se)


def test_abm_deterministic_seed_reproducibility():
    alpha = dict(OPT=1, OBS=1)
    tab, isC = abm.payoff_table(alpha, af.DEFAULTS)
    c1 = abm.simulate(tab, isC, 100, 5, 1.0, 0.005, 50_000, 5_000, 1050)
    c2 = abm.simulate(tab, isC, 100, 5, 1.0, 0.005, 50_000, 5_000, 1050)
    assert c1 == c2
    assert 0.5 < c1 <= 1.0
