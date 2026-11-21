"""Render Main Figures 1–5 and Extended Data Figures 1–7 from canonical
node-level data and recomputed summary statistics.

Preserves all figure layouts, panel labels (a, b, c, d), typography, colour
palettes, and plotted quantities of the manuscript identically.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd

from . import FIGURES_OUT_DIR, PRECOMPUTED_DIR, RESULTS_OUT_DIR, affordance as af, summary_stats

BASE, SEC, TICK = 7, 6, 5.5
W2 = 7.2
PRIM_LABEL = {
    "OPT": "Exit option", "OBS": "Contributions observed", "ANTI": "Antisocial targeting",
    "LTG": "Loners targetable", "DAG": "Defectors sanction", "LAG": "Loners sanction",
    "RET": "Response stage", "rHI": "r = 4 (vs 3)", "bHI": "β = 3 (vs 1)",
    "wHI": "w = 10 (vs 1)", "MHI": "M = 1,000 (vs 100)",
}
PRIM_COL = {
    "OPT": "#4C72B0", "OBS": "#8172B3", "ANTI": "#C44E52", "LTG": "#DD8452",
    "DAG": "#937860", "LAG": "#55A868", "RET": "#8C8C8C",
}
FAM_LABEL = {"punish": "Peer punishment", "reward": "Peer reward", "exclude": "Ostracism", "pool": "Pool punishment"}
KER_LABEL = {"uniform": "Uniform", "action": "Action-balanced", "trait": "Single-trait (reset)", "redraw": "Single-trait (redraw)"}
KER_MARK = {"uniform": "o", "action": "s", "trait": "^", "redraw": "D"}
NAMED = {"H4": "1100000", "GT9": "1110100", "GT12": "1110110", "GT17": "1111100", "RN24": "1111110"}
NSHORT = {"H4": "4", "GT9": "9", "GT12": "12", "GT17": "17", "RN24": "24"}
DISP = ["LTG", "DAG", "LAG", "ANTI"]
DSHORT = {"LTG": "Loners\ntargetable", "DAG": "Defectors\nsanction", "LAG": "Loners\nsanction", "ANTI": "Antisocial\ntargeting"}
COND = ["ANTI", "LTG", "DAG", "LAG", "RET"]


def _configure_mpl() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": BASE,
        "axes.titlesize": BASE,
        "axes.labelsize": BASE,
        "legend.fontsize": SEC,
        "xtick.labelsize": TICK,
        "ytick.labelsize": TICK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "legend.frameon": False,
        "axes.titlelocation": "left",
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    })


def _letter(ax, s: str, dx: float = -0.16, dy: float = 1.04) -> None:
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=BASE + 2, fontweight="bold", va="bottom", ha="left")


def _save(fig, out_dir: Path, name: str, generated: list[Path]) -> None:
    pdf_p = out_dir / f"{name}.pdf"
    png_p = out_dir / f"{name}.png"
    fig.savefig(pdf_p, bbox_inches="tight")
    fig.savefig(png_p, bbox_inches="tight", dpi=300)
    plt.close(fig)
    generated.extend([pdf_p, png_p])


def render_all_figures(input_dir: Path | None = None,
                       results_dir: Path | None = None,
                       figures_dir: Path | None = None) -> list[Path]:
    """Render Fig1–Fig5 and ED_Fig1–ED_Fig7 as PDF and 300-dpi PNG files.
    If summary files (`paper_numbers.json`, `sweepB_attrib.csv`, `selection_scan.csv`,
    `mc_shapley.csv`) are not yet present in `results_dir`, they are computed first
    from `input_dir` so `reproduce_figures.py` works standalone on a fresh clone."""
    _configure_mpl()
    inp = Path(input_dir) if input_dir is not None else PRECOMPUTED_DIR
    res = Path(results_dir) if results_dir is not None else RESULTS_OUT_DIR
    fig_dir = Path(figures_dir) if figures_dir is not None else FIGURES_OUT_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)
    res.mkdir(parents=True, exist_ok=True)

    needed = ["paper_numbers.json", "sweepB_attrib.csv", "selection_scan.csv", "mc_shapley.csv"]
    if not all((res / fn).exists() for fn in needed):
        summary_stats.compute_paper_numbers(input_dir=inp, results_out_dir=res)

    with open(res / "paper_numbers.json", encoding="utf-8") as fh:
        N = json.load(fh)
    lat = pd.concat([pd.read_csv(inp / f, dtype={"node": str})
                     for f in ["lattice_reference.csv", "lattice_reference_redraw.csv"]])
    val = N["validation"]
    swB = pd.read_csv(res / "sweepB_attrib.csv")
    scan = pd.read_csv(res / "selection_scan.csv")
    mc = pd.read_csv(res / "mc_shapley.csv")
    pool = pd.read_csv(inp / "htg_pool.csv", index_col=0)
    tests = N["htg"]
    REF = lat[(lat.tag == "ref") & (lat.kernel == "uniform")]

    generated: list[Path] = []

    # =========================================================================
    # Fig. 1  calculus and validation
    # =========================================================================
    fig = plt.figure(figsize=(W2, 2.7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 1.0, 1.0], wspace=0.55)
    ax = fig.add_subplot(gs[0]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    boxes = [
        (0.02, 0.70, "7 binary primitive\naffordances α", "#EAEAF2"),
        (0.52, 0.70, "Compiler under\na stated\npolicy grammar", "#EAEAF2"),
        (0.52, 0.36, "Strategy set S(α),\npayoffs, mutation\nkernel κ", "#EAEAF2"),
        (0.02, 0.36, "Rare-mutation\nstationary\ndistribution", "#EAEAF2"),
        (0.02, 0.02, "Assumption map\nv(α) over 2⁷ nodes", "#FBE9E7"),
        (0.52, 0.02, "Shapley values,\ninteractions and\ndividends", "#FBE9E7"),
    ]
    for x, y, t, c in boxes:
        ax.add_patch(FancyBboxPatch((x, y), 0.44, 0.26, boxstyle="round,pad=0.01,rounding_size=0.03", fc=c, ec="0.35", lw=0.6))
        ax.text(x + 0.22, y + 0.13, t, ha="center", va="center", fontsize=SEC)
    arr = dict(arrowstyle="-|>", lw=0.7, color="0.25", mutation_scale=7)
    for (x0, y0), (x1, y1) in [((0.46, 0.83), (0.52, 0.83)), ((0.74, 0.70), (0.74, 0.62)),
                               ((0.52, 0.49), (0.46, 0.49)), ((0.24, 0.36), (0.24, 0.28)),
                               ((0.46, 0.15), (0.52, 0.15))]:
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=arr)
    ax.set_title("Assumptions are compiled,\nnot hand-picked", fontsize=BASE, x=0.06)
    _letter(ax, "a", dx=-0.04, dy=1.06)

    ax = fig.add_subplot(gs[1])
    P = REF[REF.family == "punish"].copy(); P["k"] = P[af.PRIMS].sum(1)
    rng = np.random.default_rng(3)
    ax.scatter(P.k + rng.uniform(-0.18, 0.18, len(P)), P.nstrat, s=7, c="0.65", lw=0, zorder=2)
    for key, node in NAMED.items():
        r = P[P.node == node].iloc[0]
        ax.scatter(r.k, r.nstrat, s=22, facecolor="#C44E52", edgecolor="k", lw=0.4, zorder=3)
        ax.annotate(NSHORT[key], (r.k, r.nstrat), xytext=(5, -2), textcoords="offset points", fontsize=SEC)
    ax.set_yscale("log"); ax.set_yticks([2, 5, 10, 20, 48]); ax.set_yticklabels(["2", "5", "10", "20", "48"])
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_xlabel("Primitive affordances switched on"); ax.set_ylabel("Strategies compiled")
    ax.set_title(f"Published sets are 5 of\n{N['distinct_games']['punish']} distinct games", fontsize=BASE); ax.margins(0.06)
    _letter(ax, "b")

    ax = fig.add_subplot(gs[2])
    lim = pd.DataFrame(val["limits"]); lim = lim[lim.w == 1000.0]
    cols = {"GT9": "#4C72B0", "GT12": "#55A868", "GT17": "#DD8452", "RN24": "#C44E52"}
    for nd in ["GT9", "GT12", "GT17", "RN24"]:
        g = lim[lim.node == nd]
        ax.plot(g.M, g.max_abs_err, "-o", ms=3, lw=0.9, color=cols[nd], label=f"{NSHORT[nd]} strategies")
    Ms = np.array([100, 10000]); ax.plot(Ms, 5.0 / Ms, ls=":", color="0.5", lw=0.8)
    ax.text(1700, 5.0 / 1700 * 1.4, "slope −1", fontsize=SEC, color="0.4")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks([100, 1000, 10000]); ax.set_xticklabels(["100", "1k", "10k"])
    ax.set_xlabel("Population size M"); ax.set_ylabel("Max |abundance − M→∞ limit|")
    ax.set_title("Nodes approach published\nlimits (w = 1,000)", fontsize=BASE)
    ax.legend(loc="lower left", handlelength=1.2)
    _letter(ax, "c")
    _save(fig, fig_dir, "Fig1", generated)

    # =========================================================================
    # Fig. 2  assumption map (peer punishment)
    # =========================================================================
    rows_p = ["OPT", "OBS", "ANTI"]; cols_p = ["LTG", "DAG", "LAG", "RET"]
    rk = list(itertools.product([0, 1], repeat=3)); ck = list(itertools.product([0, 1], repeat=4))

    def grid(df, metric):
        Lf = df.set_index("node"); G = np.zeros((8, 16))
        for i, rb in enumerate(rk):
            for j, cb in enumerate(ck):
                a = dict(zip(rows_p, rb)); a.update(dict(zip(cols_p, cb)))
                G[i, j] = Lf.loc[af.node_key(a), metric]
        return G

    def bitlabels(keys):
        return ["".join("●" if b else "○" for b in k) for k in keys]

    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.9), gridspec_kw=dict(wspace=0.42))
    for ax, metric, cmap, title, lab in [
        (axs[0], "coop", "viridis", "Stationary cooperation", "Cooperation (stationary)"),
        (axs[1], "anti", "magma", "Strategies that can punish cooperators", "Frequency (stationary)"),
    ]:
        G = grid(REF[REF.family == "punish"], metric)
        im = ax.imshow(G, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.set_yticks(range(8)); ax.set_yticklabels(bitlabels(rk), fontsize=TICK, family="DejaVu Sans")
        ax.set_xticks(range(16)); ax.set_xticklabels(bitlabels(ck), rotation=90, fontsize=TICK, family="DejaVu Sans")
        if metric == "coop":
            ax.set_ylabel("Exit · Observed · Antisocial")
        ax.set_xlabel("Loners targetable · Defectors sanction · Loners sanction · Response")
        for key, node in NAMED.items():
            a = dict(zip(af.PRIMS, map(int, node)))
            i = rk.index(tuple(a[p] for p in rows_p)); j = ck.index(tuple(a[p] for p in cols_p))
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="w", lw=1.2))
            ax.text(j, i, NSHORT[key], ha="center", va="center", fontsize=SEC - 1, color="w", fontweight="bold")
        ax.axhline(3.5, color="w", lw=0.6, ls="--")
        ax.set_title(title, fontsize=BASE)
        cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02); cb.set_label(lab, fontsize=SEC); cb.ax.tick_params(labelsize=TICK)
        for s in ax.spines.values():
            s.set_visible(False)
    _letter(axs[0], "a", dx=-0.22); _letter(axs[1], "b", dx=-0.22)
    _save(fig, fig_dir, "Fig2", generated)

    # =========================================================================
    # Fig. 3  attributing the disagreement (uniform kernel)
    # =========================================================================
    E = N["ref"]["ref"]["punish:uniform"]
    fig = plt.figure(figsize=(W2, 2.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.2, 0.95], wspace=0.55)
    ax = fig.add_subplot(gs[0])
    start, end = E["endpoints"]
    labs = ["4-strategy set", "Loners targetable", "Defectors sanction", "Loners sanction", "Antisocial targeting", "24-strategy set"]
    ys = np.arange(6)[::-1]
    ax.barh(ys[0], start, color="0.6", height=0.65); ax.text(start + 0.02, ys[0], f"{start:.3f}", va="center", fontsize=SEC)
    lev = start
    for i, k in enumerate(DISP, start=1):
        ax.barh(ys[i], E["phi"][k], left=lev, color=PRIM_COL[k], height=0.65)
        ax.text(lev + 0.02, ys[i], f"{E['phi'][k]:+.3f}", va="center", fontsize=SEC)
        lev += E["phi"][k]
    ax.barh(ys[5], end, color="0.3", height=0.65); ax.text(end + 0.02, ys[5], f"{end:.3f}", va="center", fontsize=SEC)
    ax.set_yticks(ys); ax.set_yticklabels(labs, fontsize=TICK); ax.set_xlabel("Cooperation"); ax.set_xlim(0, 1.1)
    ax.set_title("Shapley allocation of the\n4-versus-24 difference", fontsize=BASE)
    _letter(ax, "a", dx=-0.62)

    ax = fig.add_subplot(gs[1])
    g = swB[(swB.fam == "punish") & (swB["var"] == 0) & (swB.kernel == "uniform")]
    for i, k in enumerate(DISP):
        v = g[f"phi_{k}"].values
        parts = ax.violinplot(v, positions=[i], widths=0.75, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(PRIM_COL[k]); pc.set_alpha(0.55); pc.set_edgecolor("none")
        q = np.percentile(v, [2.5, 25, 50, 75, 97.5])
        ax.vlines(i, q[0], q[4], color="k", lw=0.5); ax.vlines(i, q[1], q[3], color="k", lw=2)
        ax.scatter(i, q[2], s=10, color="w", edgecolor="k", lw=0.5, zorder=3)
    ax.axhline(0, color="0.5", lw=0.5, ls=":")
    ax.set_xticks(range(4)); ax.set_xticklabels([DSHORT[k] for k in DISP], fontsize=TICK)
    ax.set_ylabel("Shapley value φ (cooperation)")
    sB = N["sweepB"]["punish:0:uniform"]
    ax.set_title(f"Loner targetability most negative in\n{100*sB['mostneg']['LTG']:.0f}% of {sB['n']:,} parameter sets (design B)", fontsize=BASE)
    _letter(ax, "b", dx=-0.22)

    ax = fig.add_subplot(gs[2])
    IDX = ["ANTI", "DAG", "LTG", "LAG"]
    Imat = np.zeros((4, 4))
    for i in range(4):
        for j in range(i + 1, 4):
            Imat[i, j] = Imat[j, i] = E["I"][f"{IDX[i]}x{IDX[j]}"]
    order = [IDX.index(k) for k in DISP]; Imat = Imat[np.ix_(order, order)]
    Im = np.ma.array(Imat, mask=np.triu(np.ones_like(Imat, bool))); vm = np.abs(Imat).max()
    im = ax.imshow(Im, cmap="RdBu_r", vmin=-vm, vmax=vm)
    for i in range(4):
        for j in range(4):
            if i > j:
                ax.text(j, i, f"{Imat[i, j]:+.2f}", ha="center", va="center", fontsize=SEC,
                        color="w" if abs(Imat[i, j]) > 0.6 * vm else "k")
    short = ["Loners\ntarg.", "Def.\nsanction", "Loners\nsanction", "Anti-\nsocial"]
    ax.set_xticks(range(4)); ax.set_xticklabels(short, fontsize=TICK)
    ax.set_yticks(range(4)); ax.set_yticklabels(short, fontsize=TICK)
    ax.set_xlim(-0.5, 2.5); ax.set_ylim(3.5, 0.5)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03); cb.set_label("Shapley interaction", fontsize=SEC); cb.ax.tick_params(labelsize=TICK)
    ax.set_title("Loner targetability acts\nthrough defector sanctions", fontsize=BASE)
    _letter(ax, "c", dx=-0.3)
    _save(fig, fig_dir, "Fig3", generated)

    # =========================================================================
    # Fig. 4  the mutation kernel is itself a load-bearing assumption
    # =========================================================================
    fig = plt.figure(figsize=(W2, 2.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.15, 1.0], wspace=0.5)
    ax = fig.add_subplot(gs[0])
    kers = af.ALL_KERNELS
    for j, ker in enumerate(kers):
        e = N["ref"]["ref"][f"punish:{ker}"]; en = N["ref"]["neutral"][f"punish:{ker}"]
        for i, k in enumerate(DISP):
            x = i + (j - 1.5) * 0.18
            ax.scatter(x, e["phi"][k], marker=KER_MARK[ker], s=16, color=PRIM_COL[k], edgecolor="k", lw=0.3, zorder=3)
            ax.scatter(x, en["phi"][k], marker=KER_MARK[ker], s=12, facecolor="w", edgecolor=PRIM_COL[k], lw=0.7, zorder=2)
    ax.axhline(0, color="0.5", lw=0.5, ls=":")
    ax.set_xticks(range(4)); ax.set_xticklabels([DSHORT[k] for k in DISP], fontsize=TICK)
    ax.set_ylabel("Shapley value φ (cooperation)")
    h = [Line2D([], [], marker=KER_MARK[k], ls="", color="0.3", ms=3.5, label=KER_LABEL[k]) for k in kers]
    h += [Line2D([], [], marker="o", ls="", mfc="w", mec="0.3", ms=3.5, label="Neutral (w = 0)")]
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, -0.10), ncol=5, fontsize=TICK,
               handletextpad=0.3, columnspacing=1.2, frameon=False)
    ax.set_title("Attribution at reference parameters\nunder four mutation kernels", fontsize=BASE)
    _letter(ax, "a", dx=-0.25)

    ax = fig.add_subplot(gs[1])
    width = 0.2
    for j, ker in enumerate(kers):
        e = N["sweepB"][f"punish:0:{ker}"]
        for i, k in enumerate(DISP):
            ax.bar(i + (j - 1.5) * width, 100 * e["mostneg"][k], width=width * 0.9, color=PRIM_COL[k],
                   alpha=[1.0, 0.75, 0.5, 0.3][j], edgecolor="k", lw=0.3)
    ax.set_xticks(range(4)); ax.set_xticklabels([DSHORT[k] for k in DISP], fontsize=TICK - 1)
    ax.set_ylabel("Parameter sets where most negative (%)")
    ax.set_title("Most negative primitive by kernel\n(design B; bars: kernels as in legend)", fontsize=BASE)
    _letter(ax, "b", dx=-0.25)

    ax = fig.add_subplot(gs[2])
    for ker in kers:
        g = scan[scan.kernel == ker].sort_values("w")
        ax.plot(g.w, g.phi_LTG, marker=KER_MARK[ker], ms=2.2, lw=0.9, color=PRIM_COL["LTG"],
                alpha=[1, .75, .5, .35][kers.index(ker)], label=KER_LABEL[ker])
    ax.axhline(0, color="0.5", lw=0.5, ls=":")
    ax.set_xscale("log"); ax.set_xlabel("Selection intensity w"); ax.set_ylabel("φ(loners targetable)")
    ax.set_title("Value of loner targetability\nacross selection intensities", fontsize=BASE)
    _letter(ax, "c", dx=-0.28)
    _save(fig, fig_dir, "Fig4", generated)

    # =========================================================================
    # Fig. 5  mechanism specificity; affordances vs parameters
    # =========================================================================
    fig = plt.figure(figsize=(W2, 2.75))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.75)
    ax = fig.add_subplot(gs[0])
    M = np.array([[N["sweepB"][f"{f}:0:uniform"]["cond_median"][k] for k in COND] for f in af.FAMILIES])
    vm = np.abs(M).max()
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vm, vmax=vm, aspect="auto")
    for i in range(4):
        for j in range(5):
            ax.text(j, i, f"{M[i, j]:+.3f}", ha="center", va="center", fontsize=SEC,
                    color="w" if abs(M[i, j]) > 0.55 * vm else "k")
    ax.set_xticks(range(5)); ax.set_xticklabels([PRIM_LABEL[k] for k in COND], fontsize=TICK, rotation=30, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(4)); ax.set_yticklabels([FAM_LABEL[f] for f in af.FAMILIES])
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02); cb.set_label("Median φ (cooperation)", fontsize=SEC); cb.ax.tick_params(labelsize=TICK)
    ax.set_title("Within the exit-and-observability subcube the\npivotal primitive differs by mechanism", fontsize=BASE)
    _letter(ax, "a", dx=-0.3)

    ax = fig.add_subplot(gs[1])
    phi9 = N["ext9_phi"]
    order = sorted(phi9, key=lambda k: phi9[k]); y = np.arange(len(order))
    colors = [PRIM_COL.get(k, "#2F2F2F") for k in order]
    ax.hlines(y, 0, [phi9[k] for k in order], color=colors, lw=1.2)
    for yy, k in zip(y, order):
        if k in PRIM_COL:
            ax.scatter(phi9[k], yy, s=16, color=PRIM_COL[k], zorder=3)
        else:
            ax.scatter(phi9[k], yy, s=18, facecolor="w", edgecolor="#2F2F2F", lw=0.8, zorder=4)
    ax.axvline(0, color="0.5", lw=0.5)
    ax.set_yticks(y); ax.set_yticklabels([PRIM_LABEL[k] for k in order], fontsize=TICK)
    ax.set_xlabel("Shapley value φ (cooperation)")
    ax.scatter([], [], s=16, c="0.4", label="Affordance"); ax.scatter([], [], s=18, facecolor="w", edgecolor="#2F2F2F", label="Parameter choice")
    ax.legend(loc="upper left", handletextpad=0.3)
    ax.set_title("Affordances versus selected parameter\nchoices (2⁹ nodes, exit and observability on)", fontsize=BASE)
    ax.margins(x=0.08, y=0.03)
    _letter(ax, "b", dx=-0.6, dy=1.08)
    _save(fig, fig_dir, "Fig5", generated)

    # =========================================================================
    # Extended Data Figs 1–7
    # =========================================================================
    # ED1  selection intensity: named nodes and attribution (uniform kernel)
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.3), gridspec_kw=dict(wspace=0.35))
    ax = axs[0]
    cc = {"H4": "#4C72B0", "GT9": "#8172B3", "GT12": "#55A868", "GT17": "#DD8452", "RN24": "#C44E52"}
    g = scan[scan.kernel == "uniform"].sort_values("w")
    for k in NAMED:
        ax.plot(g.w, g[f"coop_{k}"], lw=1, color=cc[k])
        if k != "H4":
            ax.text(g.w.iloc[-1] * 1.15, g[f"coop_{k}"].iloc[-1], "4, 9" if k == "GT9" else NSHORT[k],
                    fontsize=SEC, va="center", color=cc[k])
    ax.set_xscale("log"); ax.set_xlabel("Selection intensity w"); ax.set_ylabel("Cooperation")
    ax.set_title("Cooperation at the published nodes", fontsize=BASE); _letter(ax, "a")
    ax = axs[1]
    for k in DISP:
        ax.plot(g.w, g[f"phi_{k}"], lw=1, color=PRIM_COL[k], label=PRIM_LABEL[k])
    ax.axhline(0, color="0.5", lw=0.5, ls=":")
    ax.set_xscale("log"); ax.set_xlabel("Selection intensity w"); ax.set_ylabel("Shapley value φ (cooperation)")
    ax.set_title("Attribution of the 4-versus-24 difference (reference parameters)", fontsize=BASE)
    ax.legend(loc="lower left", fontsize=TICK); _letter(ax, "b")
    _save(fig, fig_dir, "ED_Fig1", generated)

    # ED2  antisocial-capable strategies: conditional attribution by family
    fig, ax = plt.subplots(figsize=(W2 * 0.62, 2.0))
    M = np.array([[N["sweepB"][f"{f}:0:uniform"]["cond_anti_median"][k] for k in COND] for f in af.FAMILIES])
    vm = np.abs(M).max()
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vm, vmax=vm, aspect="auto")
    for i in range(4):
        for j in range(5):
            ax.text(j, i, f"{M[i, j]:+.3f}", ha="center", va="center", fontsize=SEC,
                    color="w" if abs(M[i, j]) > 0.55 * vm else "k")
    ax.set_xticks(range(5)); ax.set_xticklabels([PRIM_LABEL[k] for k in COND], fontsize=TICK, rotation=25, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(4)); ax.set_yticklabels([FAM_LABEL[f] for f in af.FAMILIES])
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("Median φ (antisocial-capable)", fontsize=SEC)
    ax.set_title("Median Shapley values for antisocial-capable strategies", fontsize=BASE)
    _save(fig, fig_dir, "ED_Fig2", generated)

    # ED3  Monte Carlo Shapley
    fig, axs = plt.subplots(1, 2, figsize=(W2 * 0.75, 2.1), gridspec_kw=dict(wspace=0.45))
    gm = mc.groupby("nperm")
    ax = axs[0]
    ax.plot(gm.rmse.mean().index, gm.rmse.mean(), "-o", ms=3, color="#4C72B0", lw=1, label="RMSE vs exact")
    ax.plot(gm.mean_se.mean().index, gm.mean_se.mean(), "--", color="0.4", lw=0.8, label="Mean reported SE")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Sampled permutations"); ax.set_ylabel("Error")
    ax.legend(loc="lower left"); ax.set_title("Error falls as 1/√permutations", fontsize=BASE); _letter(ax, "a", dx=-0.25)
    ax = axs[1]
    cov = [N["mc"][str(k)]["cov"] if str(k) in N["mc"] else N["mc"][k]["cov"] for k in gm.coverage.mean().index]
    ci = [N["mc"][str(k)]["cov_ci"] if str(k) in N["mc"] else N["mc"][k]["cov_ci"] for k in gm.coverage.mean().index]
    xs = gm.coverage.mean().index.values
    ax.errorbar(xs, cov, yerr=[np.array(cov) - np.array([c[0] for c in ci]), np.array([c[1] for c in ci]) - np.array(cov)],
                fmt="-o", ms=3, color="#55A868", lw=1, elinewidth=0.6, capsize=1.5)
    ax.axhline(0.95, color="0.5", lw=0.5, ls=":")
    ax.set_xscale("log"); ax.set_ylim(0.75, 1.0); ax.set_xlabel("Sampled permutations"); ax.set_ylabel("Coverage of nominal 95% interval")
    ax.set_title("Coverage on the 2¹¹-node test lattice", fontsize=BASE); _letter(ax, "b", dx=-0.25)
    _save(fig, fig_dir, "ED_Fig3", generated)

    # ED4  maps for the other families
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.4), gridspec_kw=dict(wspace=0.12))
    for ax, fam in zip(axs, ["reward", "exclude", "pool"]):
        G = grid(REF[REF.family == fam], "coop")
        im = ax.imshow(G, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(16)); ax.set_xticklabels(bitlabels(ck), rotation=90, fontsize=TICK - 1, family="DejaVu Sans")
        ax.set_yticks(range(8)); ax.set_yticklabels(bitlabels(rk) if fam == "reward" else [], fontsize=TICK, family="DejaVu Sans")
        ax.set_title(FAM_LABEL[fam], fontsize=BASE)
        for s in ax.spines.values():
            s.set_visible(False)
    axs[0].set_ylabel("Exit · Observed · Antisocial")
    axs[1].set_xlabel("Loners targetable · Defectors sanction · Loners sanction · Response")
    cb = fig.colorbar(im, ax=axs, fraction=0.02, pad=0.01); cb.set_label("Cooperation", fontsize=SEC)
    _save(fig, fig_dir, "ED_Fig4", generated)

    # ED5  cross-society consistency check
    fig, axs = plt.subplots(1, 4, figsize=(W2, 2.3), gridspec_kw=dict(wspace=0.55, width_ratios=[1, 1, 1, 0.85]))
    short_city = {"Dnipropetrovs'k": "Dnipro", "St. Gallen": "St Gallen"}
    ax = axs[0]
    ax.scatter(pool.RoL, pool.AS_1, s=12, c="#C44E52", edgecolor="k", lw=0.3, zorder=3)
    for c_, r in pool.iterrows():
        if c_ in ["Muscat", "Samara", "Minsk", "Istanbul", "Athens", "Seoul", "Chengdu"]:
            ax.annotate(short_city.get(c_, c_), (r.RoL, r.AS_1), xytext=(3, 2), textcoords="offset points", fontsize=TICK - 0.5)
    t = tests["P1_AS1_RoL"]; ax.set_title(f"P1: τ = {t['tau']:.2f}, P = {t['p']:.3f}", fontsize=SEC)
    ax.set_xlabel("Rule of law"); ax.set_ylabel("Antisocial punishment, period 1"); ax.margins(0.08); _letter(ax, "a", dx=-0.35)

    ax = axs[1]
    ax.scatter(pool.AS_all, pool.benefit, s=12, c="#4C72B0", edgecolor="k", lw=0.3, zorder=3)
    for c_, r in pool.iterrows():
        if c_ in ["Muscat", "Athens", "Riyadh", "Boston", "Minsk", "Samara"]:
            ax.annotate(short_city.get(c_, c_), (r.AS_all, r.benefit), xytext=(3, 2), textcoords="offset points", fontsize=TICK - 0.5)
    ax.axhline(0, color="0.5", lw=0.5, ls=":")
    t = tests["P4_benefit_ASall"]; ax.set_title(f"P4: τ = {t['tau']:.2f}, P = {t['p']:.4f}", fontsize=SEC)
    ax.set_xlabel("Antisocial punishment, all periods"); ax.set_ylabel("Cooperation gain (P − N)"); ax.margins(0.08); _letter(ax, "b", dx=-0.35)

    ax = axs[2]
    ax.scatter(pool.RoL, pool.ret_slope, s=12, c="0.35", edgecolor="k", lw=0.3, zorder=3)
    ax.errorbar(pool.RoL, pool.ret_slope, yerr=1.96 * pool.ret_slope_se, fmt="none", ecolor="0.6", lw=0.6)
    t = tests["P5_retslope_RoL"]; ax.set_title(f"P5: τ = {t['tau']:.2f}, P = {t['p']:.2f}", fontsize=SEC)
    ax.set_xlabel("Rule of law"); ax.set_ylabel("Retaliation slope"); ax.margins(0.08); _letter(ax, "c", dx=-0.35)

    ax = axs[3]
    keys = [("P1_AS1_RoL", "P1"), ("P2_ASall_RoL", "P2"), ("P3_benefit_RoL", "P3"), ("P4_benefit_ASall", "P4"), ("P5_retslope_RoL", "P5")]
    for i, (k, lab) in enumerate(keys):
        t = tests[k]; yv = len(keys) - 1 - i
        col = "#2F2F2F" if k != "P5_retslope_RoL" else "0.55"
        ax.hlines(yv, t["ci95_boot"][0], t["ci95_boot"][1], color=col, lw=1); ax.scatter(t["tau"], yv, s=14, color=col, zorder=3)
        tc = tests["sensitivity"]["country_" + k]["tau"]
        ax.scatter(tc, yv - 0.25, s=10, marker="s", facecolor="w", edgecolor=col, lw=0.7, zorder=3)
    ax.axvline(0, color="0.5", lw=0.5)
    ax.set_yticks(range(len(keys))); ax.set_yticklabels([k[1] for k in keys][::-1], fontsize=SEC)
    ax.set_xlabel("Kendall τ"); ax.set_xlim(-1, 1)
    ax.scatter([], [], s=14, color="#2F2F2F", label="16 pools (95% CI)")
    ax.scatter([], [], s=10, marker="s", facecolor="w", edgecolor="#2F2F2F", label="15 countries")
    ax.legend(loc="upper center", fontsize=TICK - 0.5, bbox_to_anchor=(0.5, -0.22), frameon=False)
    _letter(ax, "d", dx=-0.35)
    _save(fig, fig_dir, "ED_Fig5", generated)

    # ED6  finite mutation
    f_abm = inp / "abm_finite_mutation.csv"
    if f_abm.exists():
        ab = pd.read_csv(f_abm, dtype={"node": str})
        ref = REF[REF.family == "punish"].set_index("node").coop
        fig, axs = plt.subplots(1, 2, figsize=(W2 * 0.8, 2.3), gridspec_kw=dict(wspace=0.45))
        ax = axs[0]
        mus = sorted(ab.mu.unique())
        cm = plt.get_cmap("viridis")
        for j, mu in enumerate(mus):
            g = ab[ab.mu == mu].groupby("node").coop.agg(["mean", "min", "max"])
            x = ref.loc[g.index].values
            ax.errorbar(x, g["mean"], yerr=[g["mean"] - g["min"], g["max"] - g["mean"]],
                        fmt="o", ms=2.5, color=cm(j / max(1, len(mus) - 1)), lw=0.5, label=f"μ = {mu:g}")
        ax.plot([0, 1], [0, 1], color="0.5", lw=0.5, ls=":")
        ax.set_xlabel("Cooperation, rare-mutation limit"); ax.set_ylabel("Cooperation, simulation")
        ax.legend(loc="upper left", fontsize=TICK); ax.set_title("16 dispute nodes: simulation versus limit", fontsize=BASE)
        _letter(ax, "a", dx=-0.25)

        ax = axs[1]
        E0 = N["ref"]["ref"]["punish:uniform"]["phi"]
        xs = np.arange(4)
        ax.bar(xs - 0.3, [E0[k] for k in DISP], width=0.15, color="0.2", label="μ → 0 (exact)")
        for j, mu in enumerate(mus):
            ph = N["abm"]["mu" + format(mu, "g").replace(".", "p")]["phi"]
            ax.bar(xs - 0.15 + 0.15 * j, [ph[k] for k in DISP], width=0.15, color=cm(j / max(1, len(mus) - 1)), label=f"μ = {mu:g}")
        ax.axhline(0, color="0.5", lw=0.5)
        ax.set_xticks(xs); ax.set_xticklabels([DSHORT[k] for k in DISP], fontsize=TICK); ax.set_ylabel("Shapley value φ (cooperation)")
        ax.legend(loc="lower right", fontsize=TICK - 0.5); ax.set_title("Allocation at finite mutation rates", fontsize=BASE)
        _letter(ax, "b", dx=-0.25)
        _save(fig, fig_dir, "ED_Fig6", generated)

    # ED7  grammar, ostracism convention, group size
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.2), gridspec_kw=dict(wspace=0.45))
    for ax, (tag, fam, title) in zip(axs, [
        ("ref_cnt", "punish", "Count-contingent grammar\n(peer punishment)"),
        ("ref_n4", "punish", "Groups of four\n(peer punishment)"),
        ("ref_void", "exclude", "Ostracism: exclusion void\nwhen all are excluded"),
    ]):
        base = N["ref"]["ref"][f"{fam}:uniform"]["phi"]; alt = N["ref"][tag][f"{fam}:uniform"]["phi"]
        xs = np.arange(4)
        ax.bar(xs - 0.18, [base[k] for k in DISP], width=0.34, color=[PRIM_COL[k] for k in DISP], alpha=0.45, label="Baseline")
        ax.bar(xs + 0.18, [alt[k] for k in DISP], width=0.34, color=[PRIM_COL[k] for k in DISP], edgecolor="k", lw=0.4, label="Variant")
        ax.axhline(0, color="0.5", lw=0.5)
        ax.set_xticks(xs); ax.set_xticklabels([DSHORT[k] for k in DISP], fontsize=TICK - 0.5)
        ax.set_title(title, fontsize=BASE)
    axs[0].set_ylabel("Shapley value φ (cooperation)")
    axs[2].legend(loc="lower left", fontsize=TICK)
    for a_, l_ in zip(axs, "abc"):
        _letter(a_, l_, dx=-0.25)
    _save(fig, fig_dir, "ED_Fig7", generated)

    return generated
