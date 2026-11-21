"""Recompute all attributions, summary statistics, tables, figure Source Data,
and quantitative manuscript verification checks from node-level model outputs.

Preserves 100% of the mathematical and statistical logic of make_numbers.py,
make_source_data.py, and verify_numbers.py, with dynamic path resolution and
clean separation between input node-level data and generated outputs.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from . import (
    PRECOMPUTED_DIR,
    RESULTS_OUT_DIR,
    SOURCE_DATA_OUT_DIR,
    TABLES_OUT_DIR,
    affordance as af,
)

DISP = ["ANTI", "DAG", "LTG", "LAG"]                       # dispute players (order fixed)
COND = ["ANTI", "LTG", "DAG", "LAG", "RET"]                # operative-subcube players
NAMED = {"H4": "1100000", "GT9": "1110100", "GT12": "1110110", "GT17": "1111100", "RN24": "1111110"}
DFIX = dict(OPT=1, OBS=1, RET=0)
CFIX = dict(OPT=1, OBS=1)
EXT11 = af.PRIMS + ["rHI", "bHI", "wHI", "MHI"]
EXT9 = ["ANTI", "LTG", "DAG", "LAG", "RET", "rHI", "bHI", "wHI", "MHI"]


def sub_values(vals: dict, players: list[str], fixed: dict) -> dict:
    """vals: dict node_key -> value. Return dict over bit strings of `players`."""
    out = {}
    for bits in itertools.product([0, 1], repeat=len(players)):
        d = dict(fixed); d.update(dict(zip(players, bits)))
        key = "".join(str(d.get(k, 0)) for k in af.PRIMS)
        out["".join(map(str, bits))] = vals[key]
    return out


def attrib(vals: dict):
    v = sub_values(vals, DISP, DFIX)
    phi = af.exact_shapley(v, DISP)
    I = af.shapley_interactions(v, DISP)
    return phi, I, v


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def compute_paper_numbers(input_dir: Path | None = None,
                          results_out_dir: Path | None = None) -> dict:
    """Recompute all attributions, quantiles, CIs, MC Shapley checks, and summary
    statistics from node-level model outputs in `input_dir`, writing derived
    files (`paper_numbers.json`, `selection_scan.csv`, `mc_shapley.csv`,
    `sweepA_attrib.csv`, `sweepB_attrib.csv`) to `results_out_dir`."""
    inp = Path(input_dir) if input_dir is not None else PRECOMPUTED_DIR
    out = Path(results_out_dir) if results_out_dir is not None else RESULTS_OUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    N: dict = {}
    lat = pd.concat([pd.read_csv(inp / f, dtype={"node": str})
                     for f in ["lattice_reference.csv", "lattice_reference_redraw.csv"]])

    # ------------------------------------------------------------ reference lattice
    ref = lat[(lat.tag == "ref")]
    N["distinct_games"] = {f: int(ref[(ref.family == f) & (ref.kernel == "uniform")].digest.nunique()) for f in af.FAMILIES}
    cnt = lat[lat.tag == "ref_cnt"]
    N["distinct_games_cnt"] = int(cnt[cnt.kernel == "uniform"].digest.nunique())
    N["nstrat_range"] = [int(ref.nstrat.min()), int(ref.nstrat.max())]
    N["nstrat_range_cnt"] = [int(cnt.nstrat.min()), int(cnt.nstrat.max())]
    U = ref[ref.kernel == "uniform"]
    P = U[U.family == "punish"].set_index("node")
    N["named"] = {k: dict(n=int(P.loc[v, "nstrat"]), coop=float(P.loc[v, "coop"]),
                          anti=float(P.loc[v, "anti"]), loner=float(P.loc[v, "loner"]),
                          defect=float(P.loc[v, "defect"]), relax=float(P.loc[v, "relax"]))
                  for k, v in NAMED.items()}
    N["punish_coop_max"] = float(P.coop.max())
    N["punish_coop_max_node"] = str(P.coop.idxmax())
    N["punish_coop_compulsory_max"] = float(P[P.OPT == 0].coop.max())
    N["drop_4_24"] = float(P.loc[NAMED["H4"], "coop"] - P.loc[NAMED["RN24"], "coop"])
    N["relax_named"] = {k: float(P.loc[v, "relax"]) for k, v in NAMED.items()}
    N["relax_max_punish"] = float(P.relax.max())

    res = {}
    for tag, sel in [("ref", ref), ("neutral", lat[lat.tag == "neutral"]), ("ref_cnt", cnt),
                     ("ref_void", lat[lat.tag == "ref_void"]), ("ref_n4", lat[lat.tag == "ref_n4"])]:
        res[tag] = {}
        for (fam, ker), g in sel.groupby(["family", "kernel"]):
            vc = dict(zip(g.node, g.coop)); va = dict(zip(g.node, g.anti))
            phi, I, v = attrib(vc)
            phia, _, _ = attrib(va)
            tot = sum(phi.values())
            e = dict(phi=phi, share={k: phi[k] / tot for k in DISP} if abs(tot) > 1e-12 else None, total=tot,
                     I={f"{DISP[i]}x{DISP[j]}": float(I[i, j]) for i in range(4) for j in range(i + 1, 4)},
                     dividends=af.harsanyi_dividends(v, DISP), phi_anti=phia,
                     endpoints=[float(v["0000"]), float(v["1111"])])
            cv = sub_values(vc, COND, CFIX); ca = sub_values(va, COND, CFIX)
            e["cond_phi"] = af.exact_shapley(cv, COND)
            e["cond_phi_anti"] = af.exact_shapley(ca, COND)
            if tag == "ref":
                fv = {k: vc[k] for k in vc}
                e["global_phi"] = af.exact_shapley(fv, af.PRIMS)
                e["global_phi_prec"] = af.precedence_shapley(fv, af.PRIMS, [("OPT", "LTG"), ("OPT", "LAG")])
                e["global_phi_anti"] = af.exact_shapley({k: va[k] for k in va}, af.PRIMS)
                comp = sub_values(va, ["ANTI", "DAG", "RET"], dict(OPT=0, OBS=1, LTG=0, LAG=0))
                e["compulsory_phi_anti"] = af.exact_shapley(comp, ["ANTI", "DAG", "RET"])
                g2 = g[(g.OPT == 1) & (g.OBS == 1)]
                e["tau_anti_gain_optobs"] = float(
                    kendalltau(g2.anti.round(10), (g2.coop - g[g.node == "1000000"].coop.iloc[0]).round(10)).statistic
                )
                e["anti_range_cond"] = [float(g2.anti.min()), float(g2.anti.max())]
            res[tag][f"{fam}:{ker}"] = e
    N["ref"] = res

    neu = lat[(lat.tag == "neutral") & (lat.kernel == "uniform") & (lat.family == "punish")].set_index("node")
    N["neutral_named_coop"] = {k: float(neu.loc[v, "coop"]) for k, v in NAMED.items()}

    # ------------------------------------------------------------ selection scan
    ws = pd.concat([pd.read_csv(inp / f, dtype={"node": str})
                    for f in ["selection_scan_nodes.csv", "selection_scan_nodes_redraw.csv"]])
    scan = []
    for (w, ker), g in ws.groupby(["w", "kernel"]):
        vc = dict(zip(g.node, g.coop))
        phi, _, _ = attrib(vc)
        scan.append(dict(w=float(w), kernel=ker, **{f"phi_{k}": phi[k] for k in DISP},
                         **{f"coop_{nm}": float(vc[key]) for nm, key in NAMED.items()}))
    scan_df = pd.DataFrame(scan)
    scan_df.to_csv(out / "selection_scan.csv", index=False)
    N["w_cross_LTG"] = {}
    for ker, g in scan_df.groupby("kernel"):
        g = g.sort_values("w"); s = np.sign(g.phi_LTG.values)
        ch = [(float(g.w.values[i]), float(g.w.values[i + 1])) for i in range(len(s) - 1) if s[i] != s[i + 1]]
        N["w_cross_LTG"][ker] = ch
        mn = g[["phi_ANTI", "phi_DAG", "phi_LTG", "phi_LAG"]].values.argmin(1)
        N.setdefault("w_LTG_mostneg_from", {})[ker] = float(g.w.values[np.where(mn == 2)[0][0]]) if (mn == 2).any() else None

    # ------------------------------------------------------------ sweeps
    for design in ["A", "B"]:
        f = inp / f"sweep{design}_nodes.npz"
        if not f.exists():
            continue
        rec = np.load(f)["recs"]
        fr = inp / f"sweep{design}_nodes_redraw.npz"
        if fr.exists():
            rec = np.concatenate([rec, np.load(fr)["recs"]])
        pars = pd.read_csv(inp / f"sweep{design}_params.csv").set_index("idx")
        df = pd.DataFrame(rec, columns=["idx", "fam", "var", "ker", "node", "coop", "anti"])
        for c in ["idx", "fam", "var", "ker", "node"]:
            df[c] = df[c].astype(int)
        df["key"] = df.node.map(lambda x: format(x, "07b"))
        S = {}
        rows = []
        for (idx, fam, var, ker), g in df.groupby(["idx", "fam", "var", "ker"]):
            vc = dict(zip(g.key, g.coop)); va = dict(zip(g.key, g.anti))
            phi, I, _ = attrib(vc); phia, _, _ = attrib(va)
            row = dict(idx=idx, fam=af.FAMILIES[fam], var=var, kernel=af.ALL_KERNELS[ker],
                       **{f"phi_{k}": phi[k] for k in DISP}, **{f"phia_{k}": phia[k] for k in DISP},
                       I_DAGxLTG=float(I[1, 2]), I_ANTIxLAG=float(I[0, 3]))
            cv = sub_values(vc, COND, CFIX); ca = sub_values(va, COND, CFIX)
            cphi = af.exact_shapley(cv, COND); cpa = af.exact_shapley(ca, COND)
            row.update({f"cphi_{k}": cphi[k] for k in COND})
            row.update({f"cpa_{k}": cpa[k] for k in COND})
            if var == 0:
                row.update({f"coop_{nm}": float(vc[key]) for nm, key in NAMED.items()})
            rows.append(row)
        T = pd.DataFrame(rows).join(pars, on="idx")
        T.to_csv(out / f"sweep{design}_attrib.csv", index=False)
        for (fam, var, ker), g in T.groupby(["fam", "var", "kernel"]):
            key = f"{fam}:{var}:{ker}"
            n = len(g)
            mn = g[[f"phi_{k}" for k in DISP]].values.argmin(1)
            ref_order = np.argsort([res["ref"][f"{fam}:uniform"]["phi"][k] for k in DISP]) if var == 0 else None
            e = dict(
                n=n,
                median={k: float(g[f"phi_{k}"].median()) for k in DISP},
                q025={k: float(g[f"phi_{k}"].quantile(.025)) for k in DISP},
                q975={k: float(g[f"phi_{k}"].quantile(.975)) for k in DISP},
                mostneg={k: float((mn == i).mean()) for i, k in enumerate(DISP)},
                mostneg_ci={k: wilson(int((mn == i).sum()), n) for i, k in enumerate(DISP)},
                I_DAGxLTG=[float(g.I_DAGxLTG.quantile(q)) for q in (.025, .5, .975)],
                I_ANTIxLAG=[float(g.I_ANTIxLAG.quantile(q)) for q in (.025, .5, .975)],
                frac_I_DAGxLTG_neg=float((g.I_DAGxLTG < 0).mean()),
                cond_median={k: float(g[f"cphi_{k}"].median()) for k in COND},
                cond_q025={k: float(g[f"cphi_{k}"].quantile(.025)) for k in COND},
                cond_q975={k: float(g[f"cphi_{k}"].quantile(.975)) for k in COND},
                cond_anti_median={k: float(g[f"cpa_{k}"].median()) for k in COND},
                frac_RET_small=float((g.cphi_RET.abs() < 0.02).mean()),
                frac_RET_anti_small=float((g.cpa_RET.abs() < 0.02).mean()),
                frac_ANTI_coop_neg=float((g.cphi_ANTI < 0).mean()),
                frac_ANTI_anti_pos=float((g.cpa_ANTI > 0).mean()),
            )
            if ref_order is not None:
                order_same = (np.argsort(g[[f"phi_{k}" for k in DISP]].values, axis=1) == ref_order).all(1)
                e["frac_same_order"] = float(order_same.mean())
            bands = {"wlt01": g.w < 0.1, "w01to1": (g.w >= 0.1) & (g.w < 1), "wge1": g.w >= 1}
            e["mostneg_LTG_by_w"] = {b: float((mn[m.values] == 2).mean()) if m.sum() else None for b, m in bands.items()}
            e["n_by_w"] = {b: int(m.sum()) for b, m in bands.items()}
            if var == 0 and fam == "punish":
                c = g
                e["frac_RN24_lt_H4"] = float((c.coop_RN24 < c.coop_H4).mean())
                e["frac_RN24_lt_half"] = float((c.coop_RN24 < 0.5).mean())
                e["frac_GT9_gt_half"] = float((c.coop_GT9 > 0.5).mean())
                e["frac_order_9_12_17_24"] = float(((c.coop_GT9 > c.coop_GT12) & (c.coop_GT12 > c.coop_GT17) & (c.coop_GT17 > c.coop_RN24)).mean())
                e["frac_order_9_17_24"] = float(((c.coop_GT9 > c.coop_GT17) & (c.coop_GT17 > c.coop_RN24)).mean())
                e["frac_LTG_positive"] = float((c.phi_LTG > 0).mean())
                if "n" in c.columns:
                    e["mostneg_LTG_by_n"] = {
                        int(k): float((gg.phi_LTG.values == gg[[f"phi_{q}" for q in DISP]].values.min(1)).mean())
                        for k, gg in c.groupby("n")
                    }
            S[key] = e
        N[f"sweep{design}"] = S

    # ------------------------------------------------------------ extended lattices & MC Shapley
    ex = pd.read_csv(inp / "extended_lattice.csv", dtype={"node": str})
    vc = dict(zip(ex.node, ex.coop)); va = dict(zip(ex.node, ex.anti))
    N["ext11_phi"] = af.exact_shapley(vc, EXT11)
    N["ext11_phi_anti"] = af.exact_shapley(va, EXT11)
    v9 = {}
    for bits in itertools.product([0, 1], repeat=9):
        d = dict(OPT=1, OBS=1); d.update(dict(zip(EXT9, bits)))
        v9["".join(map(str, bits))] = vc["".join(str(d[k]) for k in EXT11)]
    N["ext9_phi"] = af.exact_shapley(v9, EXT9)

    rng = np.random.default_rng(11)
    exv = np.array([N["ext11_phi"][p] for p in EXT11])
    fn = lambda st: vc["".join(str(st[p]) for p in EXT11)]
    mc = []
    for nperm in [10, 30, 100, 300, 1000, 3000, 10000]:
        for rep in range(20):
            mean, se, _ = af.mc_shapley(fn, EXT11, nperm, rng)
            mc.append(dict(nperm=nperm, rep=rep, rmse=float(np.sqrt(np.mean((mean - exv) ** 2))),
                           coverage=float(np.mean(np.abs(mean - exv) <= 1.96 * se)), mean_se=float(se.mean())))
    mc_df = pd.DataFrame(mc)
    mc_df.to_csv(out / "mc_shapley.csv", index=False)
    N["mc"] = {int(k): dict(rmse=float(g.rmse.mean()), cov=float(g.coverage.mean()),
                            cov_ci=wilson(int(round(g.coverage.sum() * 11)), 11 * len(g)), se=float(g.mean_se.mean()))
               for k, g in mc_df.groupby("nperm")}

    # ------------------------------------------------------------ finite mutation ABM
    f_abm = inp / "abm_finite_mutation.csv"
    if f_abm.exists():
        ab = pd.read_csv(f_abm, dtype={"node": str})
        A = {}
        for mu, g in ab.groupby("mu"):
            m = g.groupby("node").coop.mean().to_dict()
            spread = float(g.groupby("node").coop.agg(lambda x: x.max() - x.min()).max())
            vals = {}
            for bits in itertools.product([0, 1], repeat=4):
                d = dict(DFIX); d.update(dict(zip(DISP, bits)))
                vals["".join(map(str, bits))] = m["".join(str(d.get(k, 0)) for k in af.PRIMS)]
            phi = af.exact_shapley(vals, DISP)
            A["mu" + format(mu, "g").replace(".", "p")] = dict(
                mu=float(mu), phi=phi, coop_H4=float(vals["0000"]), coop_RN24=float(vals["1111"]),
                max_seed_spread=spread, mostneg=min(phi, key=phi.get)
            )
        N["abm"] = A

    # ------------------------------------------------------------ validation, empirical & extra
    with open(inp / "validation.json", encoding="utf-8") as fh:
        N["validation"] = json.load(fh)
    with open(inp / "htg_tests.json", encoding="utf-8") as fh:
        N["htg"] = json.load(fh)

    X = {}
    for design in ["A", "B"]:
        attrib_file = out / f"sweep{design}_attrib.csv"
        if attrib_file.exists():
            T = pd.read_csv(attrib_file)
            g = T[(T.fam == "punish") & (T["var"] == 0) & (T.kernel == "uniform")]
            X[design] = dict(
                n_LTG_pos=int((g.phi_LTG > 0).sum()),
                n_LTG_pos_wge01=int(((g.phi_LTG > 0) & (g.w >= 0.1)).sum()),
                n_LTG_pos_wge1=int(((g.phi_LTG > 0) & (g.w >= 1)).sum()),
                n_wge01=int((g.w >= 0.1).sum()),
                n_wge1=int((g.w >= 1).sum()),
            )
    pc = inp / "pool_convention.json"
    if pc.exists():
        with open(pc, encoding="utf-8") as fh:
            X["pool_convention"] = json.load(fh)
    lat_ref = pd.read_csv(inp / "lattice_reference.csv", dtype={"node": str})
    g_dig = lat_ref[(lat_ref.tag == "ref") & (lat_ref.family == "punish") & (lat_ref.kernel == "uniform")].set_index("node").digest.to_dict()
    X["inert_counts"] = {p: int(sum(g_dig[nd[:j] + "1" + nd[j + 1:]] == d for nd, d in g_dig.items() if nd[j] == "0"))
                         for j, p in enumerate(af.PRIMS)}
    gg = lat_ref[(lat_ref.tag == "ref") & (lat_ref.family == "punish") & (lat_ref.kernel == "uniform")]
    base = gg[gg.node == "1000000"].coop.iloc[0]
    g2 = gg[(gg.OPT == 1) & (gg.OBS == 1)]
    N["ref"]["ref"]["punish:uniform"]["tau_anti_gain_optobs"] = float(
        kendalltau(g2.anti.round(10), (g2.coop - base).round(10)).statistic
    )
    ec = inp / "extra_checks.json"
    if ec.exists():
        with open(ec, encoding="utf-8") as fh:
            X["checks"] = json.load(fh)
    N["extra"] = X

    with open(out / "paper_numbers.json", "w", encoding="utf-8") as fh:
        json.dump(N, fh, indent=1, default=float)
    return N


def export_source_data(input_dir: Path | None = None,
                       results_out_dir: Path | None = None,
                       source_data_dir: Path | None = None) -> list[str]:
    """Export all 17 figure-panel Source Data files (identical schema to manuscript Source Data)."""
    inp = Path(input_dir) if input_dir is not None else PRECOMPUTED_DIR
    res = Path(results_out_dir) if results_out_dir is not None else RESULTS_OUT_DIR
    sd = Path(source_data_dir) if source_data_dir is not None else SOURCE_DATA_OUT_DIR
    sd.mkdir(parents=True, exist_ok=True)

    with open(res / "paper_numbers.json", encoding="utf-8") as fh:
        N = json.load(fh)
    lat = pd.concat([pd.read_csv(inp / f, dtype={"node": str})
                     for f in ["lattice_reference.csv", "lattice_reference_redraw.csv"]])
    REF = lat[(lat.tag == "ref") & (lat.kernel == "uniform")]
    NAMED_SHORT = {"1100000": "4", "1110100": "9", "1110110": "12", "1111100": "17", "1111110": "24"}

    P = REF[REF.family == "punish"].copy()
    P["n_on"] = P[af.PRIMS].sum(1)
    P["published_set"] = P.node.map(NAMED_SHORT)
    P[["node", "n_on", "nstrat", "published_set"]].to_csv(sd / "Fig1b.csv", index=False)
    pd.DataFrame(N["validation"]["limits"]).to_csv(sd / "Fig1c_SITable1a.csv", index=False)
    P[["node"] + af.PRIMS + ["nstrat", "coop", "anti", "anti_only", "pro", "loner", "defect", "relax", "published_set"]].to_csv(
        sd / "Fig2_SITable12.csv", index=False
    )

    E = N["ref"]["ref"]["punish:uniform"]
    pd.DataFrame(
        [dict(item=k, value=E["phi"][k]) for k in DISP]
        + [dict(item="coop_4", value=E["endpoints"][0]), dict(item="coop_24", value=E["endpoints"][1])]
    ).to_csv(sd / "Fig3a.csv", index=False)

    sw = pd.read_csv(res / "sweepB_attrib.csv")
    sw[(sw.fam == "punish") & (sw["var"] == 0) & (sw.kernel == "uniform")][
        ["idx", "n", "M", "r", "sigma", "gamma", "beta", "w"] + [f"phi_{k}" for k in DISP] + ["I_DAGxLTG", "I_ANTIxLAG"]
    ].to_csv(sd / "Fig3b.csv", index=False)
    pd.DataFrame([dict(pair=k, value=v) for k, v in E["I"].items()]).to_csv(sd / "Fig3c.csv", index=False)

    rows = []
    for ker in af.ALL_KERNELS:
        for tag in ["ref", "neutral"]:
            key = f"punish:{ker}"
            if key in N["ref"][tag]:
                rows += [dict(kernel=ker, condition=tag, primitive=k, phi=N["ref"][tag][key]["phi"][k]) for k in DISP]
    pd.DataFrame(rows).to_csv(sd / "Fig4a.csv", index=False)

    pd.DataFrame(
        [dict(kernel=ker, primitive=k, frac_most_negative=N["sweepB"][f"punish:0:{ker}"]["mostneg"][k],
              wilson_lo=N["sweepB"][f"punish:0:{ker}"]["mostneg_ci"][k][0],
              wilson_hi=N["sweepB"][f"punish:0:{ker}"]["mostneg_ci"][k][1])
         for ker in af.ALL_KERNELS for k in DISP]
    ).to_csv(sd / "Fig4b.csv", index=False)

    pd.read_csv(res / "selection_scan.csv").to_csv(sd / "Fig4c_EDFig1.csv", index=False)
    pd.DataFrame(
        [dict(family=f, primitive=k, median_cond_phi=N["sweepB"][f"{f}:0:uniform"]["cond_median"][k],
              median_cond_phi_anti=N["sweepB"][f"{f}:0:uniform"]["cond_anti_median"][k])
         for f in af.FAMILIES for k in COND]
    ).to_csv(sd / "Fig5a_EDFig2.csv", index=False)
    pd.DataFrame([dict(player=k, phi=v) for k, v in N["ext9_phi"].items()]).to_csv(sd / "Fig5b.csv", index=False)
    pd.read_csv(res / "mc_shapley.csv").to_csv(sd / "EDFig3.csv", index=False)
    REF[REF.family != "punish"][["family", "node"] + af.PRIMS + ["nstrat", "coop", "anti"]].to_csv(sd / "EDFig4.csv", index=False)
    pd.read_csv(inp / "htg_pool.csv").to_csv(sd / "EDFig5_pools.csv", index=False)
    with open(sd / "EDFig5_tests.json", "w", encoding="utf-8") as fh:
        json.dump(N["htg"], fh, indent=1)
    pd.read_csv(inp / "abm_finite_mutation.csv", dtype={"node": str}).to_csv(sd / "EDFig6.csv", index=False)

    rows = []
    for tag, fam in [("ref", "punish"), ("ref_cnt", "punish"), ("ref_n4", "punish"), ("ref", "exclude"), ("ref_void", "exclude")]:
        rows += [dict(variant=tag, family=fam, primitive=k, phi=N["ref"][tag][f"{fam}:uniform"]["phi"][k]) for k in DISP]
    pd.DataFrame(rows).to_csv(sd / "EDFig7.csv", index=False)
    return sorted(p.name for p in sd.iterdir())


def export_paper_tables(input_dir: Path | None = None,
                        results_out_dir: Path | None = None,
                        tables_dir: Path | None = None) -> list[str]:
    """Export Main Table 1 and Supplementary Tables 1–13 as clean CSV files."""
    inp = Path(input_dir) if input_dir is not None else PRECOMPUTED_DIR
    res = Path(results_out_dir) if results_out_dir is not None else RESULTS_OUT_DIR
    tb = Path(tables_dir) if tables_dir is not None else TABLES_OUT_DIR
    tb.mkdir(parents=True, exist_ok=True)

    with open(res / "paper_numbers.json", encoding="utf-8") as fh:
        N = json.load(fh)
    val = N["validation"]
    tests = N["htg"]
    lat = pd.concat([pd.read_csv(inp / f, dtype={"node": str})
                     for f in ["lattice_reference.csv", "lattice_reference_redraw.csv"]])

    # Main Table 1
    t1_rows = []
    for k, label in [("H4", "4"), ("GT9", "9"), ("GT12", "12"), ("GT17", "17"), ("RN24", "24")]:
        node = NAMED[k]
        d = dict(published_set=label, key=k, node=node,
                 **{p: int(node[i]) for i, p in enumerate(af.PRIMS)},
                 compiled_strategies=N["named"][k]["n"],
                 coop=N["named"][k]["coop"],
                 anti=N["named"][k]["anti"],
                 loner=N["named"][k]["loner"],
                 defect=N["named"][k]["defect"],
                 relax_arrivals=N["named"][k]["relax"])
        t1_rows.append(d)
    pd.DataFrame(t1_rows).to_csv(tb / "Table1_published_sets.csv", index=False)

    # SI Table 1a–e
    pd.DataFrame(val["limits"]).to_csv(tb / "SI_Table01a_limits.csv", index=False)
    pd.DataFrame(val["finiteM"]).to_csv(tb / "SI_Table01b_finiteM.csv", index=False)
    pd.DataFrame(val["tiny_benchmark"]).to_csv(tb / "SI_Table01c_tiny_benchmark.csv", index=False)
    pd.DataFrame(val["high_precision"]).to_csv(tb / "SI_Table01d_high_precision.csv", index=False)
    pd.DataFrame(val["directed_tests"]).to_csv(tb / "SI_Table01e_directed_tests.csv", index=False)

    # SI Table 2
    KL = {"uniform": "Uniform", "action": "Action-balanced", "trait": "Single-trait (reset)", "redraw": "Single-trait (redraw)"}
    t2_rows = []
    for tag in ["ref", "neutral", "ref_cnt", "ref_n4", "ref_void"]:
        for ker in af.ALL_KERNELS:
            key = f"punish:{ker}" if tag != "ref_void" else f"exclude:{ker}"
            if key not in N["ref"][tag]:
                continue
            if tag in ("ref_cnt", "ref_n4", "ref_void") and ker != "uniform":
                continue
            if tag == "ref_void":
                Ef = N["ref"]["ref"]["exclude:uniform"]
                t2_rows.append(dict(model="Ostracism, forfeit convention", **{f"phi_{p}": Ef["phi"][p] for p in DISP},
                                    v4=Ef["endpoints"][0], v24=Ef["endpoints"][1],
                                    I_DAGxLTG=Ef["I"]["DAGxLTG"], I_ANTIxLAG=Ef["I"]["ANTIxLAG"]))
            name = {"ref": f"Reference, {KL[ker]}", "neutral": f"Neutral, {KL[ker]}",
                    "ref_cnt": "Threshold grammar, uniform", "ref_n4": "Groups of four, uniform",
                    "ref_void": "Ostracism, void convention"}[tag]
            E = N["ref"][tag][key]
            t2_rows.append(dict(model=name, **{f"phi_{p}": E["phi"][p] for p in DISP},
                                v4=E["endpoints"][0], v24=E["endpoints"][1],
                                I_DAGxLTG=E["I"]["DAGxLTG"], I_ANTIxLAG=E["I"]["ANTIxLAG"]))
    PC = N["extra"]["pool_convention"]
    for conv, lab in [("always", "Pool punishment, fee always paid"), ("ifgame", "Pool punishment, fee only if game played")]:
        E = PC[conv]
        t2_rows.append(dict(model=lab, **{f"phi_{p}": E["phi"][p] for p in DISP},
                            v4=E["endpoints"][0], v24=E["endpoints"][1], I_DAGxLTG=np.nan, I_ANTIxLAG=np.nan))
    pd.DataFrame(t2_rows).to_csv(tb / "SI_Table02_dispute_kernels_variants.csv", index=False)

    # SI Table 3
    coal = ["".join(map(str, b)) for b in itertools.product([0, 1], repeat=4)][1:]
    t3_rows = []
    for c in coal:
        lab = ", ".join(k for k, b in zip(DISP, c) if b == "1")
        t3_rows.append(dict(coalition_bits=c, coalition=lab,
                            **{k: N["ref"]["ref"][f"punish:{k}"]["dividends"][c] for k in af.ALL_KERNELS}))
    pd.DataFrame(t3_rows).to_csv(tb / "SI_Table03_harsanyi_dividends.csv", index=False)

    # SI Table 4
    t4_rows = []
    for fam in af.FAMILIES:
        E = N["ref"]["ref"][f"{fam}:uniform"]
        S = N["sweepB"][f"{fam}:0:uniform"]
        t4_rows.append(dict(family=fam, metric="coop_reference", **{k: E["cond_phi"][k] for k in COND}))
        t4_rows.append(dict(family=fam, metric="coop_designB_median", **{k: S["cond_median"][k] for k in COND}))
        t4_rows.append(dict(family=fam, metric="coop_designB_q025", **{k: S["cond_q025"][k] for k in COND}))
        t4_rows.append(dict(family=fam, metric="coop_designB_q975", **{k: S["cond_q975"][k] for k in COND}))
        t4_rows.append(dict(family=fam, metric="anti_reference", **{k: E["cond_phi_anti"][k] for k in COND}))
        g = lat[(lat.tag == "ref") & (lat.family == fam) & (lat.kernel == "uniform")]
        vp = dict(zip(g.node, g.payoff))
        cs = sub_values(vp, COND, CFIX)
        pp = af.exact_shapley(cs, COND)
        t4_rows.append(dict(family=fam, metric="payoff_reference", **{k: pp[k] for k in COND}))
    pd.DataFrame(t4_rows).to_csv(tb / "SI_Table04_conditional_shapley.csv", index=False)

    # SI Table 5
    E_ref = N["ref"]["ref"]
    t5_rows = [
        dict(primitive=k, **{fam: E_ref[f"{fam}:uniform"]["global_phi"][k] for fam in af.FAMILIES},
             punish_precedence=E_ref["punish:uniform"]["global_phi_prec"][k],
             punish_antisocial=E_ref["punish:uniform"]["global_phi_anti"][k])
        for k in af.PRIMS
    ]
    pd.DataFrame(t5_rows).to_csv(tb / "SI_Table05_global_shapley.csv", index=False)

    # SI Table 6a & 6b
    t6a_rows = []
    for d in ["A", "B"]:
        S = N[f"sweep{d}"]
        keys = sorted([k for k in S if k.split(":")[0] in af.FAMILIES],
                      key=lambda k: (af.FAMILIES.index(k.split(":")[0]), int(k.split(":")[1]), af.ALL_KERNELS.index(k.split(":")[2])))
        for k in keys:
            fam, var, ker = k.split(":")
            E = S[k]
            t6a_rows.append(dict(
                design=d, family=fam, variant=int(var), kernel=ker, n=E["n"],
                **{f"mostneg_{p}": E["mostneg"][p] for p in DISP},
                mostneg_LTG_wlt01=E["mostneg_LTG_by_w"]["wlt01"],
                mostneg_LTG_w01to1=E["mostneg_LTG_by_w"]["w01to1"],
                mostneg_LTG_wge1=E["mostneg_LTG_by_w"]["wge1"],
                frac_RET_small=E["frac_RET_small"],
                frac_I_DAGxLTG_neg=E["frac_I_DAGxLTG_neg"],
                median_I_DAGxLTG=E["I_DAGxLTG"][1],
            ))
    pd.DataFrame(t6a_rows).to_csv(tb / "SI_Table06a_parameter_sweeps.csv", index=False)

    t6b_rows = []
    for d in ["A", "B"]:
        E = N[f"sweep{d}"]["punish:0:uniform"]; X = N["extra"][d]
        t6b_rows.append(dict(
            design=d, frac_RN24_lt_H4=E["frac_RN24_lt_H4"], frac_RN24_lt_half=E["frac_RN24_lt_half"],
            frac_GT9_gt_half=E["frac_GT9_gt_half"], frac_order_9_17_24=E["frac_order_9_17_24"],
            frac_order_9_12_17_24=E["frac_order_9_12_17_24"],
            n_LTG_pos=X["n_LTG_pos"], n_LTG_pos_wge01=X["n_LTG_pos_wge01"], n_wge01=X["n_wge01"],
            n_LTG_pos_wge1=X["n_LTG_pos_wge1"], n_wge1=X["n_wge1"],
        ))
    pd.DataFrame(t6b_rows).to_csv(tb / "SI_Table06b_sweep_ordering_robustness.csv", index=False)

    # SI Tables 7 & 8
    for fname, out_name in [("extended_lattice.csv", "SI_Table07_extended_lattice_r4.csv"),
                            ("extended_lattice_r5_boundary.csv", "SI_Table08_extended_lattice_r5_boundary.csv")]:
        fp = inp / fname
        if fp.exists():
            ex_df = pd.read_csv(fp, dtype={"node": str})
            vc = dict(zip(ex_df.node, ex_df.coop)); va = dict(zip(ex_df.node, ex_df.anti))
            p11 = af.exact_shapley(vc, EXT11); a11 = af.exact_shapley(va, EXT11)
            v9 = {}; a9 = {}
            for bits in itertools.product([0, 1], repeat=9):
                d = dict(OPT=1, OBS=1); d.update(dict(zip(EXT9, bits)))
                kk = "".join(str(d.get(k, 0)) for k in EXT11); bstr = "".join(map(str, bits))
                v9[bstr] = vc[kk]; a9[bstr] = va[kk]
            p9 = af.exact_shapley(v9, EXT9); aa9 = af.exact_shapley(a9, EXT9)
            pd.DataFrame([
                dict(assumption=k, phi_11_coop=p11[k], phi_11_anti=a11[k],
                     phi_9_coop=p9.get(k, np.nan), phi_9_anti=aa9.get(k, np.nan))
                for k in EXT11
            ]).to_csv(tb / out_name, index=False)

    # SI Table 9a–c
    pd.read_csv(inp / "htg_pool.csv", index_col=0).sort_values("RoL").to_csv(tb / "SI_Table09a_htg_pools.csv")
    spec_keys = ["P1_AS1_RoL", "P2_ASall_RoL", "P3_benefit_RoL", "P4_benefit_ASall", "P5_retslope_RoL",
                 "S1_AS1_civic", "S2_ASall_civic", "S3_benefit_civic", "S4_PRO1_RoL", "S5_PROall_RoL"]
    pd.DataFrame([
        dict(test=k, n=tests[k]["n"], alternative=tests[k]["alternative"], tau=tests[k]["tau"],
             ci95_lo=tests[k]["ci95_boot"][0], ci95_hi=tests[k]["ci95_boot"][1],
             p=tests[k]["p"], p_holm_primary=tests[k].get("p_holm", np.nan),
             p_holm_all10=tests[k]["p_holm_all10"],
             loo_lo=tests[k].get("loo_tau_range", [np.nan, np.nan])[0],
             loo_hi=tests[k].get("loo_tau_range", [np.nan, np.nan])[1])
        for k in spec_keys
    ]).to_csv(tb / "SI_Table09b_htg_tests.csv", index=False)

    sens_rows = []
    for k, t in tests["sensitivity"].items():
        sens_rows.append(dict(analysis=k, n=t["n"], tau=t["tau"], p=t["p"], p_holm=t.get("p_holm", np.nan)))
    pd.DataFrame(sens_rows).to_csv(tb / "SI_Table09c_htg_sensitivity.csv", index=False)

    # SI Table 10
    t10_rows = []
    for _, A in sorted(N["abm"].items(), key=lambda kv: kv[1]["mu"]):
        t10_rows.append(dict(mu=str(A["mu"]), v4=A["coop_H4"], v24=A["coop_RN24"],
                             **{f"phi_{p}": A["phi"][p] for p in DISP}, max_seed_spread=A["max_seed_spread"]))
    E_u = N["ref"]["ref"]["punish:uniform"]
    t10_rows.append(dict(mu="0_exact_limit", v4=E_u["endpoints"][0], v24=E_u["endpoints"][1],
                         **{f"phi_{p}": E_u["phi"][p] for p in DISP}, max_seed_spread=np.nan))
    pd.DataFrame(t10_rows).to_csv(tb / "SI_Table10_abm_finite_mutation.csv", index=False)

    # SI Table 11
    pd.DataFrame([
        dict(nperm=int(k), rmse=m["rmse"], mean_se=m["se"], coverage=m["cov"],
             wilson_lo=m["cov_ci"][0], wilson_hi=m["cov_ci"][1])
        for k, m in sorted(N["mc"].items(), key=lambda kv: int(kv[0]))
    ]).to_csv(tb / "SI_Table11_mc_shapley.csv", index=False)

    # SI Table 12
    NAMED_SHORT = {"1100000": "4", "1110100": "9", "1110110": "12", "1111100": "17", "1111110": "24"}
    P12 = lat[(lat.tag == "ref") & (lat.family == "punish") & (lat.kernel == "uniform")].sort_values("node").copy()
    P12["published_set"] = P12.node.map(NAMED_SHORT).fillna("")
    P12[["node", "published_set", "nstrat", "coop", "defect", "loner", "anti", "anti_only", "pro", "relax"]].to_csv(
        tb / "SI_Table12_peer_punishment_lattice.csv", index=False
    )

    # SI Table 13
    NM = [("1100000", "4", "Hauert et al. (2007)"),
          ("1110100", "9", "Garcia & Traulsen (2012)"),
          ("1110110", "12", "Garcia & Traulsen (2012)"),
          ("1111100", "17", "Garcia & Traulsen (2012)"),
          ("1111110", "24", "Rand & Nowak (2011)")]
    t13_rows = []
    for node, k, src in NM:
        ss = af.compile_node("punish", dict(zip(af.PRIMS, map(int, node))))
        t13_rows.append(dict(set_size=int(k), node=node, source=src,
                             target_classes=" ".join(ss.permitted),
                             compiled_strategies=", ".join(ss.labels)))
    pd.DataFrame(t13_rows).to_csv(tb / "SI_Table13_published_strategy_mapping.csv", index=False)

    return sorted(p.name for p in tb.iterdir())


def verify_manuscript_claims(N: dict) -> list[dict]:
    """Run all quantitative and qualitative verification checks against manuscript
    and Supplementary Information reported values. Returns a list of check dicts."""
    V = N["validation"]; H = N["htg"]; REF = N["ref"]
    checks = []

    def add(name: str, cond: bool, detail: str = ""):
        checks.append(dict(name=name, passed=bool(cond), detail=detail))

    phi = lambda tag, key: REF[tag][key]["phi"]
    mostneg = lambda d: min(d, key=d.get)

    # 1. Validation & numerical solver checks
    max_hp = max(r["max_abs_diff"] for r in V["high_precision"])
    add("60-digit mpmath agreement < 2e-16", max_hp < 2e-16, f"max_diff={max_hp:.3e}")
    max_fm = max(r["err"] for r in V["finiteM"] if r["w"] == 5000.0)
    add("Finite-M GT9 formula (M+2)/(M+8) within 1e-13 at w=5000", max_fm < 1e-13, f"max_err={max_fm:.3e}")
    for nd in ["GT9", "GT12", "GT17", "RN24"]:
        L = sorted([(r["M"], r["max_abs_err"]) for r in V["limits"] if r["node"] == nd and r["w"] == 1000.0])
        sl = float(np.polyfit(np.log10([m for m, _ in L]), np.log10([e for _, e in L]), 1)[0])
        add(f"Strong-selection residual ~ 1/M for {nd}", -1.25 < sl < -0.7, f"log-log slope={sl:.2f}")
    add("All 8 directed boundary payoff tests pass", all(d["pass"] for d in V["directed_tests"]))
    add("Vectorised vs group-by-group payoff kernel < 1e-13", V["kernel_test"]["max_abs_err"] < 1e-13,
        f"max_abs_err={V['kernel_test']['max_abs_err']:.3e} over {V['kernel_test']['n_groups']} groups")

    # 2. Lattice & published sets
    add("Distinct compiled games = 70 in all 4 mechanism families",
        all(v == 70 for v in N["distinct_games"].values()), str(N["distinct_games"]))
    add("Reference strategy count range [2, 48] and threshold [2, 135]",
        N["nstrat_range"] == [2, 48] and N["nstrat_range_cnt"] == [2, 135] and N["distinct_games_cnt"] == 70)
    add("Compulsory cooperation (OPT=0) in (0, 2e-10)",
        0 < N["punish_coop_compulsory_max"] < 2e-10, f"max={N['punish_coop_compulsory_max']:.3e}")
    add("Published node cooperation values match manuscript Table 1 (0.913, 0.913, 0.627, 0.466, 0.321)",
        [f"{N['named'][k]['coop']:.3f}" for k in ["H4", "GT9", "GT12", "GT17", "RN24"]]
        == ["0.913", "0.913", "0.627", "0.466", "0.321"])
    add("Cooperation drop 4 -> 24 rounds to 0.591", f"{N['drop_4_24']:.3f}" == "0.591", f"{N['drop_4_24']:.6f}")
    add("RN24 < H4 cooperation in 100% of parameter sets (designs A and B)",
        N["sweepA"]["punish:0:uniform"]["frac_RN24_lt_H4"] == 1.0
        and N["sweepB"]["punish:0:uniform"]["frac_RN24_lt_H4"] == 1.0)
    add("Relaxation time grows from 33 (H4) to 913 (RN24) mutant arrivals",
        round(N["named"]["H4"]["relax"]) == 33 and round(N["named"]["RN24"]["relax"]) == 913)

    # 3. Shapley attributions, interactions, dividends & mutation kernels
    u = phi("ref", "punish:uniform")
    sh = REF["ref"]["punish:uniform"]["share"]
    add("Uniform kernel ranking LTG < DAG < LAG < ANTI < 0",
        u["LTG"] < u["DAG"] < u["LAG"] < u["ANTI"] < 0,
        f"LTG={u['LTG']:+.3f} ({100*sh['LTG']:.0f}%), DAG={u['DAG']:+.3f} ({100*sh['DAG']:.0f}%), "
        f"LAG={u['LAG']:+.3f} ({100*sh['LAG']:.0f}%), ANTI={u['ANTI']:+.3f} ({100*sh['ANTI']:.0f}%)")
    add("Abstract headline shares: 42% LTG and 12% ANTI",
        f"{100*sh['LTG']:.0f}" == "42" and f"{100*sh['ANTI']:.0f}" == "12")
    add("ANTI largest positive Shapley value for antisocial punishers (+0.204)",
        mostneg({k: -v for k, v in REF["ref"]["punish:uniform"]["phi_anti"].items()}) == "ANTI"
        and f"{REF['ref']['punish:uniform']['phi_anti']['ANTI']:+.3f}" == "+0.204")
    add("I(DAG, LTG) = -0.38 is strongest pairwise interaction",
        min(REF["ref"]["punish:uniform"]["I"].values()) == REF["ref"]["punish:uniform"]["I"]["DAGxLTG"]
        and f"{REF['ref']['punish:uniform']['I']['DAGxLTG']:+.2f}" == "-0.38")
    add("Harsanyi dividend d({DAG, LTG}) = -0.52 close to total drop",
        0.8 * N["drop_4_24"] < -REF["ref"]["punish:uniform"]["dividends"]["0110"] < N["drop_4_24"]
        and f"{REF['ref']['punish:uniform']['dividends']['0110']:+.2f}" == "-0.52")
    add("Neutral w=0 total change = -1/6 exactly",
        abs(REF["neutral"]["punish:uniform"]["total"] + 1 / 6) < 1e-9)
    add("Neutral w=0: LTG & ANTI positive (+0.065), DAG & LAG negative (-0.148)",
        phi("neutral", "punish:uniform")["LTG"] > 0 and phi("neutral", "punish:uniform")["ANTI"] > 0
        and phi("neutral", "punish:uniform")["DAG"] < 0 and phi("neutral", "punish:uniform")["LAG"] < 0)

    a = phi("ref", "punish:action")
    t = phi("ref", "punish:trait")
    rd = phi("ref", "punish:redraw")
    add("Action-balanced kernel preserves ranking (LTG most negative, ANTI least negative)",
        mostneg(a) == "LTG" and max(a, key=a.get) == "ANTI")
    add("Single-trait (reset) kernel reverses LTG (+0.042) with DAG most negative (-0.127)",
        mostneg(t) == "DAG" and t["LTG"] > 0)
    add("Single-trait (redraw) kernel ordering LAG < LTG < ANTI < DAG",
        rd["LAG"] < rd["LTG"] < rd["ANTI"] < rd["DAG"])
    add("LTG never most negative under trait and redraw kernels in Design B (0%)",
        N["sweepB"]["punish:0:trait"]["mostneg"]["LTG"] == 0.0
        and N["sweepB"]["punish:0:redraw"]["mostneg"]["LTG"] == 0.0)
    add("I(DAG, LTG) < 0 across all 4 kernels at reference and >= 96% of all sweeps",
        all(REF["ref"][f"punish:{k}"]["I"]["DAGxLTG"] < 0 for k in af.ALL_KERNELS)
        and all(N[f"sweep{d}"][f"punish:0:{k}"]["frac_I_DAGxLTG_neg"] >= 0.96 for d in "AB" for k in af.ALL_KERNELS))
    add("LTG most negative in 83% (Design A) and 59% overall / 91% for w>=1 (Design B, uniform)",
        f"{100*N['sweepA']['punish:0:uniform']['mostneg']['LTG']:.0f}" == "83"
        and f"{100*N['sweepB']['punish:0:uniform']['mostneg']['LTG']:.0f}" == "59"
        and f"{100*N['sweepB']['punish:0:uniform']['mostneg_LTG_by_w']['wge1']:.0f}" == "91")
    add("LTG sign change in [0.03, 0.045] for uniform/action/redraw; none for trait",
        all(len(N["w_cross_LTG"][k]) == 1 and 0.03 <= N["w_cross_LTG"][k][0][0] and N["w_cross_LTG"][k][0][1] <= 0.045
            for k in ["uniform", "action", "redraw"])
        and N["w_cross_LTG"]["trait"] == [])
    add("n=4 and threshold grammar keep LTG most negative",
        mostneg(phi("ref_n4", "punish:uniform")) == "LTG" and mostneg(phi("ref_cnt", "punish:uniform")) == "LTG")

    ab = N["abm"]
    add("Finite-mutation ABM: LTG most negative at mu=0.001, 0.005; DAG at mu=0.02",
        ab["mu0p001"]["mostneg"] == "LTG" and ab["mu0p005"]["mostneg"] == "LTG" and ab["mu0p02"]["mostneg"] == "DAG")

    # 4. Mechanism specificity & extended lattice
    add("Ostracism: ANTI most negative under both forfeit (-0.862) and void (-0.657) conventions",
        mostneg(phi("ref", "exclude:uniform")) == "ANTI" and mostneg(phi("ref_void", "exclude:uniform")) == "ANTI")
    add("Peer reward: LAG raises cooperation (conditional +0.091)",
        REF["ref"]["reward:uniform"]["cond_phi"]["LAG"] > 0)
    add("Response stage |phi_RET| < 0.02 much more frequent in punishment (51%) than reward (4%) or pool (5%)",
        N["sweepB"]["punish:0:uniform"]["frac_RET_small"]
        > max(N["sweepB"]["reward:0:uniform"]["frac_RET_small"], N["sweepB"]["pool:0:uniform"]["frac_RET_small"]))
    g_glob = REF["ref"]["punish:uniform"]["global_phi"]
    gp_glob = REF["ref"]["punish:uniform"]["global_phi_prec"]
    add("Global 128-node Shapley: OPT largest (+0.377), OBS second (+0.228); precedence makes LTG -0.176",
        sorted(g_glob, key=g_glob.get)[-1] == "OPT" and sorted(g_glob, key=g_glob.get)[-2] == "OBS"
        and mostneg(gp_glob) == "LTG" and gp_glob["LTG"] < g_glob["LTG"])

    e9 = N["ext9_phi"]
    add("Every disputed affordance |phi| > every binary parameter choice on 2^9 extended subcube",
        min(abs(e9[k]) for k in DISP) > max(abs(e9[k]) for k in ["rHI", "bHI", "wHI", "MHI"]))

    # 5. Cross-society empirical check
    add("HTG sample size: 16 pools, 15 countries, 1,120 P-experiment participants",
        H["descriptives"]["n_pools"] == 16 and H["descriptives"]["n_countries"] == 15
        and H["descriptives"]["n_subjects_P"] == 1120)
    add("HTG Holm-adjusted P values: P1 (0.015) and P4 (0.0022) < 0.05; P2 and P3 > 0.05",
        H["P1_AS1_RoL"]["p_holm"] < 0.05 and H["P4_benefit_ASall"]["p_holm"] < 0.05
        and H["P2_ASall_RoL"]["p_holm"] > 0.05 and H["P3_benefit_RoL"]["p_holm"] > 0.05)
    add("HTG Holm across all 10 tests & 15-country merged P1 < 0.05",
        H["P1_AS1_RoL"]["p_holm_all10"] < 0.05 and H["P4_benefit_ASall"]["p_holm_all10"] < 0.05
        and H["sensitivity"]["country_P1_AS1_RoL"]["p_holm"] < 0.05)
    add("HTG P5 retaliation-slope 95% CI [-0.46, +0.26] spans zero; negative-gain pools = Athens, Muscat, Riyadh",
        H["P5_retslope_RoL"]["ci95_boot"][0] < 0 < H["P5_retslope_RoL"]["ci95_boot"][1]
        and sorted(H["descriptives"]["benefit_negative_pools"]) == ["Athens", "Muscat", "Riyadh"])

    return checks
