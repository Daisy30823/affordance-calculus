"""Unit tests for the affordance compiler, policy grammars, lattice collapse,
and stage-game payoff evaluation."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import affordance as af, lattice_models


def test_published_strategy_sets_and_labels():
    expected = {
        "H4": (dict(OPT=1, OBS=1), 4, ["C-N", "C-P", "D-N", "L-N"]),
        "GT9": (dict(OPT=1, OBS=1, ANTI=1, DAG=1), 9, None),
        "GT12": (dict(OPT=1, OBS=1, ANTI=1, DAG=1, LAG=1), 12, None),
        "GT17": (dict(OPT=1, OBS=1, ANTI=1, DAG=1, LTG=1), 17, None),
        "RN24": (dict(OPT=1, OBS=1, ANTI=1, DAG=1, LTG=1, LAG=1), 24, None),
    }
    for name, (alpha, count, labels) in expected.items():
        ss = af.compile_node("punish", alpha)
        assert len(ss.strategies) == count, name
        if labels is not None:
            assert ss.labels == labels


def test_methods_equation1_strategy_cardinality():
    """Verify Eq. (1) of Methods holds at all 128 nodes across all 4 families."""
    for fam in af.FAMILIES:
        for alpha in af.all_nodes():
            ss = af.compile_node(fam, alpha, grammar="base")
            n_pi = len(ss.permitted)
            expected = 0
            for act in ss.actions:
                in_g = 1 if act in ss.agency else 0
                q_adm = 1 if (ss.alpha["RET"] == 1 and ss.obs_class[act] in ss.permitted and len(ss.agency) > 0) else 0
                expected += (2 ** (n_pi * in_g)) * (1 + q_adm)
            assert len(ss.strategies) == expected


def test_distinct_games_and_threshold_grammar():
    for fam in af.FAMILIES:
        digests = {af.compile_node(fam, a, "base").digest() for a in af.all_nodes()}
        assert len(digests) == 70
    cnt_sets = [af.compile_node("punish", a, "cnt") for a in af.all_nodes()]
    assert len({s.digest() for s in cnt_sets}) == 70
    assert min(len(s.strategies) for s in cnt_sets) == 2
    assert max(len(s.strategies) for s in cnt_sets) == 135
    ss_rn24_cnt = af.compile_node("punish", dict(OPT=1, OBS=1, ANTI=1, DAG=1, LTG=1, LAG=1), "cnt")
    assert len(ss_rn24_cnt.strategies) == 45


def test_digest_implies_identical_payoff_tensors():
    for fam in af.FAMILIES:
        by_digest = {}
        for alpha in af.all_nodes():
            ss = af.compile_node(fam, alpha)
            by_digest.setdefault(ss.digest(), []).append(ss)
        for lst in by_digest.values():
            p0 = af.pair_payoffs(lst[0], af.DEFAULTS)
            for ss in lst[1:]:
                p1 = af.pair_payoffs(ss, af.DEFAULTS)
                assert np.max(np.abs(p0[0] - p1[0])) == 0.0
                assert np.max(np.abs(p0[1] - p1[1])) == 0.0


def test_directed_boundary_cases():
    results = lattice_models.directed_tests()
    assert len(results) == 8
    for r in results:
        assert r["pass"], f"Failed case: {r}"


def test_vectorised_vs_member_by_member_payoffs():
    rng = np.random.default_rng(42)
    for fam in af.FAMILIES:
        for gram in ["base", "cnt"]:
            for _ in range(10):
                alpha = {k: int(rng.integers(2)) for k in af.PRIMS}
                ss = af.compile_node(fam, alpha, gram)
                S = len(ss.strategies)
                par = {
                    **af.DEFAULTS,
                    "n": int(rng.integers(3, 7)),
                    "r": float(rng.uniform(1.5, 4.5)),
                    "excl": str(rng.choice(["forfeit", "void"])),
                    "pool_fee": str(rng.choice(["always", "ifgame"])),
                }
                PX, PY = af.pair_payoffs(ss, par)
                for _ in range(8):
                    x, y = rng.integers(S, size=2)
                    a = int(rng.integers(1, par["n"]))
                    ref = af.group_payoffs(ss, [int(x)] * a + [int(y)] * (par["n"] - a), par)
                    assert abs(ref[0] - PX[x, y, a - 1]) < 1e-13
                    if x != y:
                        assert abs(ref[-1] - PY[x, y, a - 1]) < 1e-13
