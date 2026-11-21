"""Exploratory cross-society consistency check of the assumption map against the
public punishment-experiment data of Herrmann, Thoeni & Gaechter (2008)
(Dryad doi:10.5061/dryad.87301, CC0).

Pool-level statistics; Kendall tau-b; Monte Carlo permutation P values
(1e6 permutations for the main tests, 2e5 for sensitivity analyses).
The predictions below were derived from the model and written into this
script before the hypothesis tests were run. Pool-level descriptive statistics
had been inspected during data preparation, so the predictions are not
preregistered and all tests are exploratory.
  P1  antisocial punishment in period 1 decreases with rule of law     (one-sided)
  P2  antisocial punishment over all periods decreases with rule of law (one-sided)
  P3  cooperation gain from the punishment opportunity increases with rule of law (one-sided)
  P4  cooperation gain decreases with antisocial punishment             (one-sided)
  P5  retaliation slope versus rule of law: no gradient expected; tested two-sided
      (a non-significant result is inconclusive, not evidence of absence)
Antisocial punishment = punishment points assigned to group members who
contributed at least as much as the punisher (Herrmann et al.'s definition),
averaged over such eligible punisher-target pairs. Recipients learned only the
total points received, not who assigned them, so the retaliation slope is a
'blind' lagged-response measure.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import kendalltau

from . import RAW_DATA_DIR, RESULTS_OUT_DIR

NPERM = 1_000_000
NPERM_S = 200_000

SPEC = {
    "P1_AS1_RoL": ("AS_1", "RoL", "less"),
    "P2_ASall_RoL": ("AS_all", "RoL", "less"),
    "P3_benefit_RoL": ("benefit", "RoL", "greater"),
    "P4_benefit_ASall": ("benefit", "AS_all", "less"),
    "P5_retslope_RoL": ("ret_slope", "RoL", "two-sided"),
    "S1_AS1_civic": ("AS_1", "civic", "less"),
    "S2_ASall_civic": ("AS_all", "civic", "less"),
    "S3_benefit_civic": ("benefit", "civic", "greater"),
    "S4_PRO1_RoL": ("PRO_1", "RoL", "two-sided"),
    "S5_PROall_RoL": ("PRO_all", "RoL", "two-sided"),
}
PRIMARY = ["P1_AS1_RoL", "P2_ASall_RoL", "P3_benefit_RoL", "P4_benefit_ASall"]


def load(data_csv: Path | None = None) -> pd.DataFrame:
    csv_path = Path(data_csv) if data_csv is not None else (RAW_DATA_DIR / "HerrmannThoeniGaechterDATA.csv")
    d = pd.read_csv(csv_path, skiprows=3, low_memory=False)
    d = d.loc[:, ~d.columns.str.startswith("Unnamed")]
    for c in ["period", "senderscontribution", "otherscontribution", "punishment",
              "recpun", "civic", "ruleoflaw", "secseq"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def pool_table(d: pd.DataFrame):
    Pd = d[d.p == "P-experiment"].copy()
    Nd = d[d.p == "N-experiment"].copy()
    Pd["anti_opp"] = Pd.otherscontribution >= Pd.senderscontribution
    s = Pd.drop_duplicates(["subjectid", "period"])[["subjectid", "period", "recpun"]].copy()
    s["period"] += 1
    Pd = Pd.merge(s.rename(columns={"recpun": "recpun_lag"}), on=["subjectid", "period"], how="left")
    pool = Pd.groupby("city").agg(RoL=("ruleoflaw", "first"), civic=("civic", "first"))
    pool["n_subj"] = Pd.groupby("city").subjectid.nunique()
    pool["AS_all"] = Pd[Pd.anti_opp].groupby("city").punishment.mean()
    pool["AS_1"] = Pd[Pd.anti_opp & (Pd.period == 1)].groupby("city").punishment.mean()
    pool["PRO_all"] = Pd[~Pd.anti_opp].groupby("city").punishment.mean()
    pool["PRO_1"] = Pd[~Pd.anti_opp & (Pd.period == 1)].groupby("city").punishment.mean()
    cP = Pd.drop_duplicates(["subjectid", "period"]).groupby("city").senderscontribution.mean()
    cN = Nd.drop_duplicates(["subjectid", "period"]).groupby("city").senderscontribution.mean()
    pool["contrib_P"], pool["contrib_N"] = cP, cN
    pool["benefit"] = cP - cN
    slopes, ses = {}, {}
    for city, g in Pd[Pd.anti_opp & (Pd.period > 1)].groupby("city"):
        g = g.dropna(subset=["recpun_lag"])
        X = sm.add_constant(g.recpun_lag.astype(float))
        m = sm.OLS(g.punishment.astype(float), X).fit(cov_type="cluster",
                                                      cov_kwds={"groups": g.subjectid})
        slopes[city], ses[city] = m.params.iloc[1], m.bse.iloc[1]
    pool["ret_slope"], pool["ret_slope_se"] = pd.Series(slopes), pd.Series(ses)
    return pool, Pd


def tau_perm(x, y, alternative, nperm=NPERM, seed=1):
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y)); x, y = x[ok], y[ok]
    t_obs = kendalltau(x, y).statistic
    n = len(x)
    iu = np.triu_indices(n, 1)
    sx = np.sign(x[:, None] - x[None, :])[iu]
    sy_full = np.sign(y[:, None] - y[None, :])
    nx = np.sqrt((sx != 0).sum()); ny = np.sqrt((sy_full[iu] != 0).sum())
    rng = np.random.default_rng(seed)
    cnt = 0; done = 0; B = 50_000
    while done < nperm:
        b = min(B, nperm - done)
        perms = np.argsort(rng.random((b, n)), axis=1)
        sy = np.sign(y[perms][:, :, None] - y[perms][:, None, :])[:, iu[0], iu[1]]
        t = (sy * sx).sum(1) / (nx * ny)
        if alternative == "less":
            cnt += (t <= t_obs + 1e-12).sum()
        elif alternative == "greater":
            cnt += (t >= t_obs - 1e-12).sum()
        else:
            cnt += (np.abs(t) >= abs(t_obs) - 1e-12).sum()
        done += b
    p = (cnt + 1) / (nperm + 1)
    return dict(tau=float(t_obs), p=float(p), n=int(n), alternative=alternative,
                mc_se=float(np.sqrt(p * (1 - p) / nperm)))


def boot_tau(x, y, nboot=20000, seed=2, q=(2.5, 97.5)):
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y)); x, y = x[ok], y[ok]
    rng = np.random.default_rng(seed)
    n = len(x); out = []
    for _ in range(nboot):
        i = rng.integers(n, size=n)
        if np.unique(x[i]).size > 1 and np.unique(y[i]).size > 1:
            out.append(kendalltau(x[i], y[i]).statistic)
    return [float(np.percentile(out, q[0])), float(np.percentile(out, q[1]))]


def pool_variants(Pd):
    """Alternative operationalisations of antisocial punishment."""
    v = pd.DataFrame(index=sorted(Pd.city.unique()))
    strict = Pd.otherscontribution > Pd.senderscontribution
    v["AS_strict_1"] = Pd[strict & (Pd.period == 1)].groupby("city").punishment.mean()
    v["AS_strict_all"] = Pd[strict].groupby("city").punishment.mean()
    # per punisher-period (denominator independent of contribution patterns)
    tot = Pd.assign(ap=Pd.punishment * Pd.anti_opp).groupby(["city", "subjectid", "period"]).ap.sum().reset_index()
    v["AS_pc_1"] = tot[tot.period == 1].groupby("city").ap.mean()
    v["AS_pc_all"] = tot.groupby("city").ap.mean()
    return v


def holm(pdict):
    ps = sorted((p, k) for k, p in pdict.items())
    m = len(ps); adj = {}; run = 0
    for i, (p, k) in enumerate(ps):
        run = max(run, min(1.0, (m - i) * p)); adj[k] = run
    return adj


def country_table(pool):
    """Merge the two Swiss pools (shared country-level indices), weighting by participants."""
    c = pool.copy()
    c["country"] = [("Switzerland" if x in ("Zurich", "St. Gallen") else x) for x in c.index]
    num = [k for k in c.columns if k not in ("country",)]
    agg = {}
    for k in num:
        agg[k] = c.groupby("country", group_keys=False).apply(
            lambda g: np.average(g[k], weights=g["n_subj"]) if g[k].notna().all() else np.nan,
            include_groups=False,
        )
    out = pd.DataFrame(agg)
    out["n_subj"] = c.groupby("country").n_subj.sum()
    return out


def run_htg_analysis(data_csv: Path | None = None, out_dir: Path | None = None,
                     nperm: int = NPERM, nperm_s: int = NPERM_S,
                     nboot: int = 20000, verbose: bool = True) -> dict:
    out = Path(out_dir) if out_dir is not None else RESULTS_OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    d = load(data_csv)
    pool, Pd = pool_table(d)
    pool = pool.join(pool_variants(Pd))
    pool.to_csv(out / "htg_pool.csv")
    tests = {}
    for k, (a, b, alt) in SPEC.items():
        r = tau_perm(pool[a], pool[b], alt, nperm=nperm)
        r["ci95_boot"] = boot_tau(pool[a], pool[b], nboot=nboot)
        r["ci90_boot"] = boot_tau(pool[a], pool[b], nboot=nboot, q=(5, 95))
        r["x"], r["y"] = a, b
        tests[k] = r
        if verbose:
            print(f"  {k:20s} tau={r['tau']:+.3f}  p={r['p']:.4f}", flush=True)
    adj = holm({k: tests[k]["p"] for k in PRIMARY})
    for k in PRIMARY:
        tests[k]["p_holm"] = adj[k]
    adj10 = holm({k: tests[k]["p"] for k in SPEC})
    for k in SPEC:
        tests[k]["p_holm_all10"] = adj10[k]
    for k in PRIMARY:
        a, b, alt = SPEC[k]
        vals = [kendalltau(pool.drop(index=c)[a], pool.drop(index=c)[b], nan_policy="omit").statistic for c in pool.index]
        tests[k]["loo_tau_range"] = [float(min(vals)), float(max(vals))]
    # sensitivity: countries (Swiss pools merged)
    ctab = country_table(pool)
    ctab.to_csv(out / "htg_country.csv")
    sens = {}
    for k in PRIMARY + ["P5_retslope_RoL"]:
        a, b, alt = SPEC[k]
        r = tau_perm(ctab[a], ctab[b], alt, nperm=nperm_s, seed=3)
        vals = [kendalltau(ctab.drop(index=c)[a], ctab.drop(index=c)[b], nan_policy="omit").statistic for c in ctab.index]
        r["loo_country_tau_range"] = [float(min(vals)), float(max(vals))]
        sens["country_" + k] = r
    adjc = holm({k: sens["country_" + k]["p"] for k in PRIMARY})
    for k in PRIMARY:
        sens["country_" + k]["p_holm"] = adjc[k]
    # sensitivity: alternative antisocial-punishment definitions
    for tag, c1, call in [("strict", "AS_strict_1", "AS_strict_all"), ("percapita", "AS_pc_1", "AS_pc_all")]:
        for k, (a, b, alt) in [("P1", (c1, "RoL", "less")), ("P2", (call, "RoL", "less")), ("P4", ("benefit", call, "less"))]:
            sens[f"{tag}_{k}"] = tau_perm(pool[a], pool[b], alt, nperm=nperm_s, seed=4)
    tests["sensitivity"] = sens
    g = Pd[Pd.anti_opp & (Pd.period == 1)]
    X = sm.add_constant(g.ruleoflaw.astype(float))
    mdl = sm.OLS(g.punishment.astype(float), X).fit(cov_type="cluster", cov_kwds={"groups": g.city})
    tests["pair_level_period1"] = dict(coef=float(mdl.params.iloc[1]), se=float(mdl.bse.iloc[1]),
                                       n_obs=int(len(g)), n_clusters=int(g.city.nunique()),
                                       n_pairs_period1_total=int((Pd.period == 1).sum()))
    tests["descriptives"] = dict(
        n_pools=int(len(pool)), n_countries=int(len(ctab)), n_subjects_P=int(Pd.subjectid.nunique()),
        AS_all_range=[float(pool.AS_all.min()), float(pool.AS_all.max())],
        AS1_range=[float(pool.AS_1.min()), float(pool.AS_1.max())],
        cv_AS_all=float(pool.AS_all.std() / pool.AS_all.mean()),
        cv_PRO_all=float(pool.PRO_all.std() / pool.PRO_all.mean()),
        n_benefit_negative=int((pool.benefit < 0).sum()),
        benefit_negative_pools=sorted(pool.index[pool.benefit < 0].tolist()),
        tau_AS1_ASall=float(kendalltau(pool.AS_1, pool.AS_all).statistic))
    with open(out / "htg_tests.json", "w", encoding="utf-8") as fh:
        json.dump(tests, fh, indent=1)
    return tests
