"""Affordance calculus (version 2).

Compile explicit assumptions about observation, targeting rights, agency and
response into the strategy set they imply within a stated policy grammar, and
evaluate each compiled game with the rare-mutation limit of a finite-population
Moran process with exponential fitness.

Primitive affordances (binary):
  OPT  exit option: individuals may abstain (loner action L, payoff sigma)
  OBS  individual contributions are observed (C and D are distinct classes)
  ANTI the instrument may be applied to its antisocial target
       (sanctioning contributors; rewarding non-contributors)
  LTG  non-participants remain in the group's sanctioning stage and may be targeted
  DAG  non-contributing participants hold the instrument
  LAG  non-participants hold the instrument (and observe the group)
  RET  a response stage: those acted upon respond to each actor
       (counter-punishment in the sanctioning families, reciprocated reward in
       the reward family)

Policy grammar.
  base : a strategy is (action, T, q). T is a fixed subset of permitted target
         classes (acted on whenever present), q an unconditional response bit.
  cnt  : count-contingent extension. Every non-empty T additionally exists in a
         threshold version that acts on members of a target class only if at
         least two co-players belong to that class; every responding strategy
         additionally exists in a version that responds only if acted on by at
         least two co-players.

Mechanism families: 'punish', 'reward', 'exclude' (ostracism), 'pool'.
Ostracism convention for groups in which every participant is excluded:
  'forfeit' (default): nobody receives the public good, contributors still pay c;
  'void'   : exclusion is void and the good is shared among all participants.

The module depends only on numpy and scipy.
"""
from __future__ import annotations
import itertools, hashlib
from dataclasses import dataclass, field
from math import factorial
import numpy as np
from scipy.special import logsumexp
from scipy.stats import hypergeom

PRIMS = ["OPT", "OBS", "ANTI", "LTG", "DAG", "LAG", "RET"]
FAMILIES = ["punish", "reward", "exclude", "pool"]
CLASSES = ["C", "D", "P", "L"]
CIDX = {c: i for i, c in enumerate(CLASSES)}
KERNELS = ["uniform", "action", "trait"]
ALL_KERNELS = ["uniform", "action", "trait", "redraw"]
DEFAULTS = dict(M=100, n=5, r=3.0, c=1.0, sigma=1.0, gamma=0.3, beta=1.0,
                w=1.0, G=0.3, rho_R=1.0, excl="forfeit", pool_fee="always", logrho_floor=-700.0)


@dataclass(frozen=True)
class Strategy:
    action: str
    targets: frozenset
    theta: int = 1      # sanction threshold (1 = unconditional)
    ret: int = 0        # response bit
    rtheta: int = 1     # response threshold (1 = respond to any act)

    def label(self, classes):
        tg = "" if not classes else "-" + "".join("P" if k in self.targets else "N" for k in classes)
        th = "*" if self.theta == 2 else ""
        rt = ("r" + ("*" if self.rtheta == 2 else "")) if self.ret else ""
        return f"{self.action}{tg}{th}{rt}"


@dataclass
class StrategySet:
    family: str
    alpha: dict
    grammar: str
    actions: list
    obs_class: dict
    permitted: list
    agency: list
    strategies: list = field(default_factory=list)

    @property
    def labels(self):
        return [s.label(self.permitted) for s in self.strategies]

    def signature(self):
        """Canonical representation of the compiled game (strategy set and
        observation map); two profiles are the same game iff equal."""
        return (self.family, self.grammar,
                tuple(sorted((s.action, tuple(sorted(s.targets)), s.theta, s.ret, s.rtheta)
                             for s in self.strategies)),
                tuple(sorted(self.obs_class.items())))

    def digest(self):
        return hashlib.sha1(repr(self.signature()).encode()).hexdigest()[:16]


def compile_node(family: str, alpha: dict, grammar: str = "base") -> StrategySet:
    a = {k: int(alpha.get(k, 0)) for k in PRIMS}
    actions = ["C", "D"] + (["L"] if a["OPT"] else [])
    obs_class = {"C": "C" if a["OBS"] else "P", "D": "D" if a["OBS"] else "P", "L": "L"}
    if family == "reward":
        perm_actions = {"C"} | ({"D"} if a["ANTI"] else set())
    else:
        perm_actions = {"D"} | ({"C"} if a["ANTI"] else set())
    if a["OPT"] and a["LTG"]:
        perm_actions |= {"L"}
    present = []
    for act in actions:
        if obs_class[act] not in present:
            present.append(obs_class[act])
    # robust-legality convention: a class may be targeted only if every action
    # that maps to it is a permitted target
    permitted = [k for k in present if all(act in perm_actions for act in actions if obs_class[act] == k)]
    agency = ["C"] + (["D"] if a["DAG"] else []) + (["L"] if (a["OPT"] and a["LAG"]) else [])
    agency = agency if permitted else []
    cnt = grammar == "cnt"
    strategies = []
    for act in actions:
        pols = [(frozenset(), 1)]
        if act in agency:
            pols = []
            for bits in itertools.product([0, 1], repeat=len(permitted)):
                T = frozenset(k for k, b in zip(permitted, bits) if b)
                pols.append((T, 1))
                if cnt and T:
                    pols.append((T, 2))
        targetable = obs_class[act] in permitted and len(agency) > 0
        resp = [(0, 1)]
        if a["RET"] and targetable:
            resp = [(0, 1), (1, 1)] + ([(1, 2)] if cnt else [])
        for (T, th) in pols:
            for (q, rth) in resp:
                strategies.append(Strategy(act, T, th, q, rth))
    return StrategySet(family, a, grammar, actions, obs_class, permitted, agency, strategies)


def all_nodes(prims=PRIMS, fixed=None):
    fixed = fixed or {}
    free = [k for k in prims if k not in fixed]
    for bits in itertools.product([0, 1], repeat=len(free)):
        d = dict(fixed); d.update(dict(zip(free, bits)))
        yield {k: d.get(k, 0) for k in PRIMS}


def node_key(alpha):
    return "".join(str(int(alpha[k])) for k in PRIMS)


# ----------------------------------------------------------------------------
# payoffs
# ----------------------------------------------------------------------------
def _arrays(ss):
    S = len(ss.strategies)
    act = np.array([s.action for s in ss.strategies])
    cls = np.array([CIDX[ss.obs_class[s.action]] for s in ss.strategies])
    T = np.zeros((S, len(CLASSES)))
    for i, s in enumerate(ss.strategies):
        for k in s.targets:
            T[i, CIDX[k]] = 1.0
    th = np.array([s.theta for s in ss.strategies], float)
    q = np.array([s.ret for s in ss.strategies], float)
    rth = np.array([s.rtheta for s in ss.strategies], float)
    return S, act, cls, T, th, q, rth


def pair_payoffs(ss, par):
    """PX[x,y,a-1], PY[x,y,a-1]: payoffs to an X and to a Y player in a group
    with a players of X and n-a of Y, a = 1..n-1."""
    p = {**DEFAULTS, **par}
    n, r, c, sig, g, b = int(p["n"]), p["r"], p["c"], p["sigma"], p["gamma"], p["beta"]
    fam = ss.family
    S, act, cls, T, th, q, rth = _arrays(ss)
    A = np.arange(1, n)[None, None, :].astype(float)
    B = n - A
    X = lambda v: v[:, None, None]
    Y = lambda v: v[None, :, None]
    same = (cls[:, None] == cls[None, :]).astype(float)[:, :, None]
    tXX = X(T[np.arange(S), cls]); tYY = Y(T[np.arange(S), cls])
    tXY = T[:, cls][:, :, None]; tYX = np.transpose(T[:, cls])[:, :, None]
    # count-contingent thresholds (number of co-players in the target class)
    cXX = (A - 1) + B * same; cXY = B + (A - 1) * same
    cYY = (B - 1) + A * same; cYX = A + (B - 1) * same
    eXX = tXX * (cXX >= X(th)); eXY = tXY * (cXY >= X(th))
    eYY = tYY * (cYY >= Y(th)); eYX = tYX * (cYX >= Y(th))
    isC = (act == "C").astype(float); isL = (act == "L").astype(float)
    partX, partY = 1 - X(isL), 1 - Y(isL)
    actsX = (A - 1) * eXX + B * eXY
    actsY = (B - 1) * eYY + A * eYX
    hX = (A - 1) * eXX + B * eYX
    hY = (B - 1) * eYY + A * eXY
    nC = A * X(isC) + B * Y(isC)
    nP = A * partX + B * partY
    game = nP >= 2
    if fam == "exclude":
        exX = partX * (hX > 0); exY = partY * (hY > 0)
        R = A * partX * (1 - exX) + B * partY * (1 - exY)
        if p["excl"] == "void":
            void = (R == 0)
            exX = np.where(void, 0.0, exX); exY = np.where(void, 0.0, exY)
            R = np.where(void, nP, R)
        share = np.where((R > 0) & game, r * c * nC / np.maximum(R, 1), 0.0)
        baseX = np.where(game, share * (1 - exX) - c * X(isC), sig) * partX + sig * (1 - partX)
        baseY = np.where(game, share * (1 - exY) - c * Y(isC), sig) * partY + sig * (1 - partY)
    else:
        share = np.where(game, r * c * nC / np.maximum(nP, 1), 0.0)
        baseX = np.where(game, share - c * X(isC), sig) * partX + sig * (1 - partX)
        baseY = np.where(game, share - c * Y(isC), sig) * partY + sig * (1 - partY)
    if fam == "punish":
        sX = -g * actsX - b * hX; sY = -g * actsY - b * hY
    elif fam == "reward":
        sX = -g * actsX + p["rho_R"] * hX; sY = -g * actsY + p["rho_R"] * hY
    elif fam == "exclude":
        sX = -g * actsX; sY = -g * actsY
    elif fam == "pool":
        fX = X((T.sum(1) > 0).astype(float)); fY = Y((T.sum(1) > 0).astype(float))
        if p.get("pool_fee", "always") == "ifgame":     # pool contribution paid only when the game is played
            fX = fX * game; fY = fY * game
        sX = -p["G"] * fX - b * hX; sY = -p["G"] * fY - b * hY
    else:
        raise ValueError(fam)
    # response stage
    respX = X(q) * (hX >= X(rth)); respY = Y(q) * (hY >= Y(rth))
    effect = p["rho_R"] if fam == "reward" else -b        # reciprocated reward vs counter-fine
    costX = g * respX * hX; costY = g * respY * hY
    recvX = effect * ((A - 1) * eXX * respX + B * eXY * respY)
    recvY = effect * ((B - 1) * eYY * respY + A * eYX * respX)
    PX = baseX + sX - costX + recvX
    PY = baseY + sY - costY + recvY
    return np.broadcast_to(PX, (S, S, n - 1)).copy(), np.broadcast_to(PY, (S, S, n - 1)).copy()


def group_payoffs(ss, members, par):
    """Member-by-member reference evaluation of an arbitrary group."""
    p = {**DEFAULTS, **par}
    r, c, sig, g, b = p["r"], p["c"], p["sigma"], p["gamma"], p["beta"]
    st = [ss.strategies[m] for m in members]
    n = len(st)
    cls = [ss.obs_class[s.action] for s in st]
    part = [s.action != "L" for s in st]
    nC = sum(s.action == "C" for s in st)
    acts = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j or cls[j] not in st[i].targets:
                continue
            k = cls[j]
            cnt = sum(1 for l in range(n) if l != i and cls[l] == k)
            if cnt >= st[i].theta:
                acts[i, j] = 1
    h = acts.sum(0)
    pay = np.zeros(n)
    game = sum(part) >= 2
    if ss.family == "exclude":
        excl = [part[j] and h[j] > 0 for j in range(n)]
        R = sum(part[j] and not excl[j] for j in range(n))
        if R == 0 and p["excl"] == "void":
            excl = [False] * n; R = sum(part)
        share = r * c * nC / R if (R > 0 and game) else 0.0
        for i in range(n):
            pay[i] = sig if (not part[i] or not game) else (0 if excl[i] else share) - c * (st[i].action == "C")
    else:
        share = r * c * nC / sum(part) if game else 0.0
        for i in range(n):
            pay[i] = sig if (not part[i] or not game) else share - c * (st[i].action == "C")
    for i in range(n):
        if ss.family == "punish":
            pay[i] += -g * acts[i].sum() - b * h[i]
        elif ss.family == "reward":
            pay[i] += -g * acts[i].sum() + p["rho_R"] * h[i]
        elif ss.family == "exclude":
            pay[i] += -g * acts[i].sum()
        elif ss.family == "pool":
            fee = (len(st[i].targets) > 0) and (game or p.get("pool_fee", "always") == "always")
            pay[i] += -p["G"] * fee - b * h[i]
    effect = p["rho_R"] if ss.family == "reward" else -b
    for i in range(n):
        if st[i].ret and h[i] >= st[i].rtheta:
            for j in range(n):
                if acts[j, i]:
                    pay[i] -= g
                    pay[j] += effect
    return pay


def homogeneous_payoffs(ss, par):
    p = {**DEFAULTS, **par}
    n = int(p["n"])
    return np.array([group_payoffs(ss, [s] * n, p)[0] for s in range(len(ss.strategies))])


def log_fixation_matrix(ss, par):
    """L[x,y] = log fixation probability of a single X mutant among M-1 Y residents.
    Moran birth-death process, fitness exp(w*payoff), groups sampled without
    replacement (the focal individual excluded from its own co-player sample)."""
    p = {**DEFAULTS, **par}
    M, n, w = int(p["M"]), int(p["n"]), p["w"]
    PX, PY = pair_payoffs(ss, p)
    S = PX.shape[0]
    homo = homogeneous_payoffs(ss, p)
    PXf = np.concatenate([PX, np.broadcast_to(homo[:, None, None], (S, S, 1))], axis=2)
    PYf = np.concatenate([np.broadcast_to(homo[None, :, None], (S, S, 1)), PY], axis=2)
    i = np.arange(1, M); j = np.arange(0, n)
    HX = hypergeom.pmf(j[None, :], M - 1, i[:, None] - 1, n - 1)
    HY = hypergeom.pmf(j[None, :], M - 1, i[:, None], n - 1)
    piX = np.einsum("xyk,ik->xyi", PXf, HX)
    piY = np.einsum("xyk,ik->xyi", PYf, HY)
    D = -w * (piX - piY)
    Lc = np.concatenate([np.zeros((S, S, 1)), np.cumsum(D, axis=2)], axis=2)
    L = -logsumexp(Lc, axis=2)
    np.fill_diagonal(L, -np.inf)
    return L


def fixation_matrix(ss, par):
    p = {**DEFAULTS, **par}
    L = log_fixation_matrix(ss, p)
    rho = np.exp(np.maximum(L, p["logrho_floor"]))
    np.fill_diagonal(rho, 0.0)
    return rho


# ----------------------------------------------------------------------------
# mutation kernels and stationary distribution
# ----------------------------------------------------------------------------
def _genotype(ss):
    """Full genotype vectors with unexpressed genes fixed at 0."""
    acts = ss.actions
    G = []
    for s in ss.strategies:
        v = [acts.index(s.action)] + [int(k in s.targets) for k in ss.permitted] + [s.theta - 1, s.ret, s.rtheta - 1]
        G.append(v)
    return np.array(G)


def mutation_kernel(ss, kind="uniform"):
    """kappa[x, y] = probability that a mutant arising in resident y uses strategy x."""
    S = len(ss.strategies)
    if S == 1:
        return np.zeros((1, 1))
    if kind == "uniform":
        k = np.ones((S, S))
    elif kind == "action":
        acts = [s.action for s in ss.strategies]
        cnt = {a: acts.count(a) for a in set(acts)}
        wv = np.array([1.0 / (len(cnt) * cnt[a]) for a in acts])
        k = np.repeat(wv[:, None], S, axis=1)
    elif kind == "trait":
        # change exactly one trait; traits not expressed by an action are reset
        G = _genotype(ss)
        d = (G[:, None, :] != G[None, :, :]).sum(2)
        k = (d == 1).astype(float)
    elif kind == "redraw":
        # change one trait within the current action, or change the action and
        # draw the new action's policy uniformly
        G = _genotype(ss)
        d = (G[:, None, :] != G[None, :, :]).sum(2)
        a = G[:, 0]
        same = a[:, None] == a[None, :]
        nA = np.array([(a == a[x]).sum() for x in range(S)], float)
        k = np.where(same, (d == 1).astype(float), 1.0 / nA[:, None])
    else:
        raise ValueError(kind)
    np.fill_diagonal(k, 0.0)
    k = k / k.sum(0, keepdims=True)
    return k


def stationary(rho, kappa=None):
    """Stationary distribution of the embedded rare-mutation chain with
    off-diagonal transition weights kappa[x,y]*rho[x,y] (y -> x), computed with
    the Grassmann-Taksar-Heyman algorithm (no subtractions; stable for rates
    spanning hundreds of orders of magnitude)."""
    S = rho.shape[0]
    if S == 1:
        return np.ones(1)
    if kappa is None:
        kappa = np.full((S, S), 1.0 / (S - 1))
    Q = (kappa * rho).T.copy()            # Q[y, x]: rate y -> x
    np.fill_diagonal(Q, 0.0)
    P = Q.astype(np.longdouble)
    for k in range(S - 1, 0, -1):
        s = P[k, :k].sum()
        if s <= 0:
            s = np.finfo(np.longdouble).tiny
        P[:k, k] /= s
        P[:k, :k] += np.outer(P[:k, k], P[k, :k])
    pi = np.zeros(S, dtype=np.longdouble); pi[0] = 1.0
    for k in range(1, S):
        pi[k] = (pi[:k] * P[:k, k]).sum()
    pi = pi / pi.sum()
    return np.asarray(pi, dtype=float)


def relaxation_time(rho, kappa=None):
    """1/(1-|lambda_2|) of the embedded jump chain, in units of mutant arrivals."""
    S = rho.shape[0]
    if S == 1:
        return 0.0
    if kappa is None:
        kappa = np.full((S, S), 1.0 / (S - 1))
    Pm = (kappa * rho).T.copy(); np.fill_diagonal(Pm, 0.0)
    np.fill_diagonal(Pm, 1.0 - Pm.sum(1))
    ev = np.sort(np.abs(np.linalg.eigvals(Pm)))[::-1]
    return float(1.0 / max(1.0 - ev[1], 1e-300))


def outcomes(ss, pi, par=None):
    st = ss.strategies
    act = np.array([s.action for s in st])
    cC, cD = ss.obs_class["C"], ss.obs_class["D"]
    hitC = np.array([cC in s.targets for s in st], float)
    hitD = np.array([cD in s.targets for s in st], float)
    hitL = np.array(["L" in s.targets for s in st], float)
    if ss.family == "reward":
        anti, pro = hitD, hitC
    else:
        anti, pro = hitC, hitD
    distinct = float(cC != cD)
    out = dict(coop=float(pi[act == "C"].sum()), defect=float(pi[act == "D"].sum()),
               loner=float(pi[act == "L"].sum()), anti=float(pi @ anti), pro=float(pi @ pro),
               anti_only=float(pi @ (anti * (1 - pro))), anti_obs=float(pi @ anti) * distinct,
               hitL=float(pi @ hitL), active=float(pi @ np.array([len(s.targets) > 0 for s in st], float)),
               ret=float(pi @ np.array([s.ret for s in st], float)))
    if par is not None:
        out["payoff"] = float(pi @ homogeneous_payoffs(ss, par))
    return out


def solve(family, alpha, par=None, kernel="uniform", grammar="base", extra=False):
    par = {**DEFAULTS, **(par or {})}
    ss = compile_node(family, alpha, grammar)
    rho = fixation_matrix(ss, par)
    kap = mutation_kernel(ss, kernel)
    pi = stationary(rho, kap)
    out = outcomes(ss, pi, par)
    if extra:
        out["relax"] = relaxation_time(rho, kap)
    return ss, rho, pi, out


# ----------------------------------------------------------------------------
# attribution
# ----------------------------------------------------------------------------
def exact_shapley(v, players):
    """v maps bit strings over `players` (in order) to values; baseline = all off."""
    K = len(players)
    phi = {}
    for k, pk in enumerate(players):
        tot = 0.0
        for bits in itertools.product([0, 1], repeat=K):
            if bits[k]:
                continue
            s = sum(bits)
            wgt = factorial(s) * factorial(K - s - 1) / factorial(K)
            on = list(bits); on[k] = 1
            tot += wgt * (v["".join(map(str, on))] - v["".join(map(str, bits))])
        phi[pk] = tot
    return phi


def shapley_interactions(v, players):
    K = len(players)
    I = np.zeros((K, K))
    for a_, b_ in itertools.combinations(range(K), 2):
        tot = 0.0
        others = [k for k in range(K) if k not in (a_, b_)]
        for sub in itertools.product([0, 1], repeat=K - 2):
            s = sum(sub)
            wgt = factorial(s) * factorial(K - s - 2) / factorial(K - 1)
            base = [0] * K
            for k, bb in zip(others, sub):
                base[k] = bb
            def val(xa, xb):
                bb = list(base); bb[a_] = xa; bb[b_] = xb
                return v["".join(map(str, bb))]
            tot += wgt * (val(1, 1) - val(1, 0) - val(0, 1) + val(0, 0))
        I[a_, b_] = I[b_, a_] = tot
    return I


def harsanyi_dividends(v, players):
    """Moebius transform: d(S) = sum_{T subset S} (-1)^{|S|-|T|} v(T)."""
    K = len(players)
    out = {}
    for bits in itertools.product([0, 1], repeat=K):
        S = [i for i in range(K) if bits[i]]
        tot = 0.0
        for sub in itertools.product([0, 1], repeat=len(S)):
            tb = [0] * K
            for i, sb in zip(S, sub):
                tb[i] = sb
            tot += (-1) ** (len(S) - sum(sub)) * v["".join(map(str, tb))]
        out["".join(map(str, bits))] = tot
    return out


def precedence_shapley(v, players, precedence):
    """Shapley value averaged only over orderings that respect precedence
    constraints [(a, b), ...] meaning a must be switched on before b
    (a permission-structure value)."""
    K = len(players)
    idx = {p: i for i, p in enumerate(players)}
    tot = np.zeros(K); cnt = 0
    for order in itertools.permutations(range(K)):
        pos = {o: t for t, o in enumerate(order)}
        if any(pos[idx[a]] > pos[idx[b]] for a, b in precedence):
            continue
        bits = [0] * K
        prev = v["".join(map(str, bits))]
        for o in order:
            bits[o] = 1
            cur = v["".join(map(str, bits))]
            tot[o] += cur - prev
            prev = cur
        cnt += 1
    return {p: tot[i] / cnt for i, p in enumerate(players)}


def mc_shapley(value_fn, players, n_perm, rng):
    K = len(players)
    samples = np.zeros((n_perm, K))
    for t in range(n_perm):
        order = rng.permutation(K)
        state = {p_: 0 for p_ in players}
        prev = value_fn(state)
        for k in order:
            state[players[k]] = 1
            cur = value_fn(state)
            samples[t, k] = cur - prev
            prev = cur
    return samples.mean(0), samples.std(0, ddof=1) / np.sqrt(n_perm), samples
