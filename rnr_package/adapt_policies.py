"""
How fast to adapt: step vs ramp, seat 0, restricted responses from the solved bank.

Seat 0 identifies the two opponents (grid identifier, showdown cards, 1% tremble) and every
--every hands picks a strategy from the bank of independent-bias restricted responses:

    cell   the most likely bank cell: each grid hypothesis is assigned to its nearest bank cell
           (alpha 0 = equilibrium player; otherwise the nearest of 0.2, 0.4, 0.6, ties DOWN),
           the joint posterior is summed per cell, and the cell with the most mass wins.
    c      confidence = (P(most likely type pair) - 1/16) / (1 - 1/16), as in session_sim.py
    p*     that cell's largest safe p (worst case within 0.05 of the best equilibrium-level
           strategy; exact punisher by default, pessimistic with --pess)

    step T   p = 0 until c > T, then p*                                   (T = 0.5, 0.9)
    ramp B   p = p* * c**B, snapped DOWN to the largest solved p for that cell  (B = 1, 3)
    halfT    once c > T, p = p*/2 snapped down to a solved p (often 0 when p* is small)
    notchT   once c > T, one solved point below p*: the largest solved p < p* ("back off" a notch)
    nbhdT    once c > T, the largest solved p whose strategy is safe against EVERY cell within 0.2
             of the model in each opponent's alpha (same types) -- safe under small model error
    sel      SELECTION from the whole bank: take the smallest set of bank cells holding 90% of the
             posterior; among every bank strategy (any cell, any solved p) keep those safe against EVERY
             cell in that set; play the one with the highest posterior-weighted profit across it
             (p = 0 if none). selb: the same, with a 0.03 safety buffer (re-solves moved margins up to
             ~0.065). Needs the bank matrix: cross_eval.py --all-p.
    brT      once c > T, an UNRESTRICTED best response to the most likely cell's model (any
             cell, in the bank or not); shows what the restriction costs and what it buys
    p0       p = 0 throughout                                           (reference; exact)
    cell     the true opponents' nearest cell at its p*, from hand 1    (reference; exact)
    best     hindsight: the bank strategy (any cell, any solved p) earning most against the TRUE
             opponents while staying safe against them; shortfall is measured against this

If the most likely cell is not in the bank, the policy plays p = 0.

Measured exactly at every update: expected payoff relative to equilibrium; cumulative shortfall
against the oracle (chips over the session); and the pessimistic worst case of the strategy in
play against the true opponents. Every strategy played is a bank entry, so each (strategy, true
opponents) pair is evaluated once and cached.

    python adapt_policies.py --time-one         # time one scenario x seed, plus its evaluations
    python adapt_policies.py                    # simulate, evaluate, report (resumable)
    python adapt_policies.py --report
    python adapt_policies.py --pess             # p* from the pessimistic re-evaluation (own folder)
"""

import argparse
import glob
import json
import os
import pickle
import re
import time
from multiprocessing import Pool

import numpy as np

HET, OUT, SEAT = "results_het", "adapt", 0
GRID = (0.2, 0.4, 0.6)
TOL = 0.05
SCENARIOS = {                       # name: (seat 1 spec, seat 2 spec)
    "FR_onGrid": ("F:0.4", "R:0.2"),
    "RO_onGrid": ("R:0.4", "O"),
    "CR_onGrid": ("C:0.2", "R:0.6"),
    "FR_offGrid": ("F:0.3", "R:0.3"),
    "switch_F_to_R": ("F:0.35>R:0.3@200", "O"),
    "switch_FR_to_RR": ("F:0.4>R:0.4@150", "R:0.2"),     # two deviators throughout: speed matters
}
POLICIES = ["step0.5", "step0.9", "ramp1", "notch0.5", "nbhd0.5", "sel", "selb", "br0.5"]


# ---------------------------------------------------------------- styles and cells

def phases(spec):
    """'F:0.35>R:0.3@200' -> [(0, 'F', 0.35), (200, 'R', 0.3)];  'O' -> [(0, 'O', 0.0)]."""
    out = []
    for part in spec.split(">"):
        style, _, at = part.partition("@")
        t, _, a = style.partition(":")
        a = float(a or 0)
        out.append((int(at) if at else 0, t if a > 0 else "O", a if t != "O" else 0.0))
    return out


def style_at(spec, hand):
    return [(t, a) for start, t, a in phases(spec) if start < hand or start == 0][-1]


def nearest_cell_label(t, a):
    """A (type, alpha) hypothesis -> its bank label: equilibrium, or the nearest grid alpha (ties down)."""
    if t == "O" or a == 0:
        return ("O", 0.0)
    return (t, min(GRID, key=lambda g: (abs(g - a), g)))


def truth_key(seat1, seat2):
    (t1, a1), (t2, a2) = seat1, seat2
    return f"{t1}{a1:g}-{t2}{a2:g}"


# ---------------------------------------------------------------- the bank

def load_bank(pess=False, seat=None):
    """{(t1, a1, t2, a2): {"pstar": p*, "ps": sorted solved ps, "paths": {p: strategy}}}, one seat (default seat 0), grid only."""
    seat = SEAT if seat is None else seat
    pat = re.compile(r"rnr_([ORCF]{2})_a([0-9.]+)(?:-([0-9.]+))?_s" + str(seat) + r"_independent_selfish_p([0-9.]+)\.json$")
    eq = json.load(open(os.path.join(HET, f"linear_OO_a0.0_s{seat}.json")))["rows"][0]["vs_biased"]
    cells = {}
    for f in glob.glob(os.path.join(HET, f"rnr_*_s{seat}_independent_selfish_p*.json")):
        m = pat.search(os.path.basename(f))
        if not m:
            continue
        pair, a1 = m.group(1), float(m.group(2)); a2 = float(m.group(3)) if m.group(3) else a1
        t1, t2 = pair
        a1, a2 = (a1 if t1 != "O" else 0.0), (a2 if t2 != "O" else 0.0)
        if (a1 and a1 not in GRID) or (a2 and a2 not in GRID):
            continue
        key = (t1 if a1 else "O", a1, t2 if a2 else "O", a2)
        npy = f[:-5] + "_strategy.npy"
        c = cells.setdefault(key, {"files": {}, "paths": {}, "pair": pair, "tag": m.group(2) + (f"-{m.group(3)}" if m.group(3) else "")})
        c["files"][float(m.group(4))] = f
        if os.path.exists(npy):
            c["paths"][float(m.group(4))] = npy
    bank = {}
    for key, c in cells.items():
        if 0.0 not in c["files"] or len(c["files"]) < 2:
            continue
        lin = os.path.join(HET, f"linear_{c['pair']}_a{c['tag']}_s{seat}.json")
        if not os.path.exists(lin):
            continue
        if pess:
            pf = os.path.join("results_het_pess", f"pess_{c['pair']}_a{c['tag']}_s{seat}.json")
            if not os.path.exists(pf):
                continue
            P = json.load(open(pf))
            wb = max(P["cfr"]["worst"], P["rnr"]["0"]["worst"])
            worst = {float(p): v["worst"] for p, v in P["rnr"].items()}
        else:
            L = json.load(open(lin))["rows"][0]
            J = {p: json.load(open(f)) for p, f in c["files"].items()}
            wb = max(L["worst_selfish"], J[0.0]["worst_selfish"]) - eq
            worst = {p: j["worst_selfish"] - eq for p, j in J.items()}
        safe = [p for p, w in worst.items() if w >= wb - TOL] + [0.0]
        bank[key] = {"pstar": max(safe), "ps": sorted(c["files"]), "paths": c["paths"]}
    return bank


# ---------------------------------------------------------------- simulation (one scenario x seed x policy)

G = {}


def init_sim(pess=False, nbhd=None, matrix=None):
    """Per-process setup. pess, the neighbourhood table and the bank matrix are passed explicitly:
    on Windows, workers do not inherit globals."""
    G["pess"], G["nbhd"], G["matrix"] = pess, (nbhd or {}), (matrix or {})
    import pyspiel
    from cfr_path import find_cfr
    from tree_engine import Tree
    from session_sim import Strategies
    G["game"] = pyspiel.load_game("leduc_poker", {"players": 3})
    G["base"] = pickle.load(open(find_cfr(), "rb"))["table"]
    G["T"] = Tree(); G["S"] = Strategies(G["T"], G["base"])
    G["bank"] = load_bank(pess)
    G["tables"] = {}


def strategy_flat(key):
    """'cfr', 'p0', or 't1a1-t2a2|p' -> flat seat-0 strategy."""
    if key == "cfr":
        return G["S"].base[SEAT]
    if key.startswith("br|"):
        t1, a1, t2, a2 = cell_from_str(key[3:])
        T, S = G["T"], G["S"]
        return T.respond([None, S.tilt(1, t1, a1), S.tilt(2, t2, a2)], SEAT, T.util[:, SEAT], T.util[:, SEAT])[0]
    if key == "p0":
        any_cell = next(c for c in G["bank"].values() if 0.0 in c["paths"])
        return np.load(any_cell["paths"][0.0])            # the shared, cached p = 0 solve
    cell, p = key.split("|")
    return np.load(G["bank"][cell_from_str(cell)]["paths"][float(p)])


def cell_str(cell):
    t1, a1, t2, a2 = cell
    return f"{t1}{a1:g}-{t2}{a2:g}"


def cell_from_str(s):
    m = re.match(r"([ORCF])([0-9.]+)-([ORCF])([0-9.]+)$", s)
    return (m.group(1), float(m.group(2)), m.group(3), float(m.group(4)))


def table(key):
    if key not in G["tables"]:
        G["tables"][key] = G["S"].table(SEAT, strategy_flat(key))
    return G["tables"][key]


def opp_table(spec, seat, hand):
    t, a = style_at(spec, hand)
    k = ("opp", seat, t, a)
    if k not in G["tables"]:
        G["tables"][k] = G["S"].table(seat, G["S"].tilt(seat, t, a))
    return G["tables"][k]


def choose(gi, policy, bank):
    """(strategy key, p, cell, c) for the policy, given the identifier's posterior."""
    from q2_grid_identify import TYPES
    P = gi.posterior("showdown")
    pt = max((gi.p_types(P, x, y), x, y) for x in ["O"] + list(TYPES) for y in ["O"] + list(TYPES))
    c = min(1.0, max(0.0, (pt[0] - 1 / 16) / (1 - 1 / 16)))
    labels = sorted({nearest_cell_label(t, a) for t, a in gi.hyp})
    idx = {l: i for i, l in enumerate(labels)}
    A = np.zeros((gi.H, len(labels)))
    for h, (t, a) in enumerate(gi.hyp):
        A[h, idx[nearest_cell_label(t, a)]] = 1
    mass = A.T @ P @ A
    i, j = np.unravel_index(np.argmax(mass), mass.shape)
    cell = labels[i] + labels[j]
    if policy.startswith("sel"):
        return select(policy, labels, mass, cell, c)
    if policy.startswith("br"):
        if c > float(policy[2:]):
            return f"br|{cell_str(cell)}", 1.0, cell, c
        return "p0", 0.0, cell, c
    if cell not in bank:
        return "p0", 0.0, cell, c
    pstar, ps = bank[cell]["pstar"], bank[cell]["ps"]
    if policy.startswith(("half", "nbhd", "notch")):
        if c <= float(policy.lstrip("abcdefghijklmnopqrstuvwxyz")):
            return "p0", 0.0, cell, c
        if policy.startswith("half"):
            p = max([q for q in bank[cell]["paths"] if q <= pstar / 2 + 1e-12], default=0.0)
        elif policy.startswith("notch"):
            p = max([q for q in bank[cell]["paths"] if q < pstar - 1e-12], default=0.0)
        else:
            p = G["nbhd"].get(cell_str(cell), 0.0)
        return ("p0", 0.0, cell, c) if p == 0.0 else (f"{cell_str(cell)}|{p:g}", p, cell, c)
    if policy.startswith("step"):
        p = pstar if c > float(policy[4:]) else 0.0
    else:
        target = pstar * c ** float(policy[4:])
        p = max([q for q in ps if q <= target + 1e-12], default=0.0)
    if p == 0.0:
        return "p0", 0.0, cell, c
    return f"{cell_str(cell)}|{p:g}", p, cell, c


def select(policy, labels, mass, cell, c, level=0.9):
    """Bank selection: safe against every cell in the posterior's 90% set, best weighted profit."""
    buffer = 0.03 if policy.endswith("b") else 0.0
    Mx = G["matrix"]
    ranked = sorted(((mass[i, j], labels[i] + labels[j]) for i in range(len(labels)) for j in range(len(labels))),
                    key=lambda t: -t[0])
    plausible, tot = [], 0.0
    for m, cl in ranked:
        plausible.append((m, cell_str(cl))); tot += m
        if tot >= level:
            break
    if any(t not in Mx for _, t in plausible):          # a plausible table nobody has scored: stay at p = 0
        return "p0", 0.0, cell, c
    keys = set.intersection(*(set(Mx[t]) for _, t in plausible))
    best = None
    for k in keys:
        if all(Mx[t][k][1] >= -TOL + buffer for _, t in plausible):
            v = sum(m * Mx[t][k][0] for m, t in plausible)
            if best is None or v > best[0]:
                best = (v, k)
    if best is None or best[1] in ("p0", "cfr") or best[0] <= 0:
        return "p0", 0.0, cell, c
    return best[1], float(best[1].split("|")[1]), cell, c


def build_matrix(cache):
    """{truth: {strategy: (EV minus baseline's, worst case minus baseline's)}}, baseline = the better of
    CFR and p = 0 against that truth; from the evaluation cache."""
    by_t = {}
    for k, v in cache.items():
        st, t = k.split("||")
        by_t.setdefault(t, {})[st] = (v["ev"], v["worst"])
    out = {}
    for t, D in by_t.items():
        if "cfr" in D and "p0" in D:
            gb, wb = max(D["cfr"][0], D["p0"][0]), max(D["cfr"][1], D["p0"][1])
            out[t] = {st: (g - gb, w - wb) for st, (g, w) in D.items() if st != "cfr" and not st.startswith("br|")}
    return out


def simulate(job):
    scen, seed, policy, hands, every, *rest = job
    forget, switch = rest if rest else (1.0, 0.0)
    from q2_grid_identify import GridIdentifier
    from q2_identify import play_hand
    s1, s2 = SCENARIOS[scen]
    rng = np.random.default_rng([seed, list(SCENARIOS).index(scen)])   # same stream start for every policy
    gi = GridIdentifier(G["game"], G["base"], seat=SEAT, tremble=0.01, forget=forget, switch=switch)
    key = "p0"
    log = [dict(hand=0, key=key, p=0.0, cell=None, c=0.0)]
    for n in range(1, hands + 1):
        hist, cards, sd, _ = play_hand(G["game"], [table(key), opp_table(s1, 1, n), opp_table(s2, 2, n)], rng)
        gi.update(hist, cards, sd)
        if n % every == 0 and n < hands:
            key, p, cell, c = choose(gi, policy, G["bank"])
            log.append(dict(hand=n, key=key, p=p, cell=cell_str(cell), c=c))
    return dict(scenario=scen, seed=seed, policy=policy, hands=hands, every=every, log=log)


# ---------------------------------------------------------------- exact evaluation (cached)

def evaluate(pair):
    """(strategy key, truth key) -> {ev, worst}: exact EV and pessimistic worst case vs equilibrium."""
    from reeval_pessimistic import respond_pess
    skey, tkey = pair
    T, S = G["T"], G["S"]
    m = re.match(r"([ORCF])([0-9.]+)-([ORCF])([0-9.]+)$", tkey)
    opp = {1: S.tilt(1, m.group(1), float(m.group(2))), 2: S.tilt(2, m.group(3), float(m.group(4)))}
    sig = strategy_flat(skey)
    eq = T.ev(S.base)[SEAT]
    prof = [sig, opp[1], opp[2]]
    worst = []
    for o, other in ((1, 2), (2, 1)):
        pr = [None] * 3; pr[SEAT], pr[other] = sig, opp[other]
        worst.append(respond_pess(T, pr, o, T.util[:, o], T.util[:, SEAT], 0.01) - eq)
    return pair, {"ev": float(T.ev(prof)[SEAT] - eq), "worst": float(min(worst))}


def segments(scen, log, hands):
    """[(from hand, to hand, strategy key, truth key)] with truth changes split out."""
    s1, s2 = SCENARIOS[scen]
    marks = sorted({e["hand"] for e in log} | {start for start, _, _ in phases(s1) + phases(s2)} | {hands})
    out = []
    for lo, hi in zip(marks[:-1], marks[1:]):
        key = [e["key"] for e in log if e["hand"] <= lo][-1]
        out.append((lo, hi, key, truth_key(style_at(s1, lo + 1), style_at(s2, lo + 1))))
    return out


def cell_oracle_key(scen, hand):
    s1, s2 = SCENARIOS[scen]
    cell = nearest_cell_label(*style_at(s1, hand)) + nearest_cell_label(*style_at(s2, hand))
    b = G["bank"].get(cell)
    if b is None or b["pstar"] == 0.0:
        return "p0"
    return f"{cell_str(cell)}|{b['pstar']:g}"


def marks_of(scen):
    return sorted({0} | {s for s, _, _ in phases(SCENARIOS[scen][0]) + phases(SCENARIOS[scen][1])})


def truths_of(scen):
    s1, s2 = SCENARIOS[scen]
    return sorted({truth_key(style_at(s1, m + 1), style_at(s2, m + 1)) for m in marks_of(scen)})


def candidates():
    """Every playable bank strategy: p = 0, CFR, and each cell at each solved p > 0."""
    out = ["p0", "cfr"]
    for c, b in G["bank"].items():
        out += [f"{cell_str(c)}|{p:g}" for p in sorted(b["paths"]) if p > 0]
    return out


def neighbours(cell):
    t1, a1, t2, a2 = cell
    return [d for d in G["bank"] if (d[0], d[2]) == (t1, t2)
            and abs(d[1] - a1) <= 0.2 + 1e-9 and abs(d[3] - a2) <= 0.2 + 1e-9]


def base_worst(cache, t):
    return max(cache[f"cfr||{t}"]["worst"], cache[f"p0||{t}"]["worst"])


def best_key(cache, t):
    """Hindsight: the candidate with the highest EV against truth t among those safe against it."""
    if f"cfr||{t}" not in cache or f"p0||{t}" not in cache:
        return "p0"                                  # not evaluated yet (report-only on a fresh cache)
    bw = base_worst(cache, t)
    ok = [(cache[f"{k}||{t}"]["ev"], k) for k in candidates()
          if f"{k}||{t}" in cache and cache[f"{k}||{t}"]["worst"] >= bw - TOL]
    return max(ok)[1] if ok else "p0"


def reference_log(scen, which, hands, every, cache=None):
    if which == "p0":
        return [dict(hand=0, key="p0")]
    if which == "cell":
        return [dict(hand=m, key=cell_oracle_key(scen, m + 1)) for m in marks_of(scen)]
    s1, s2 = SCENARIOS[scen]
    return [dict(hand=m, key=best_key(cache, truth_key(style_at(s1, m + 1), style_at(s2, m + 1))))
            for m in marks_of(scen)]


def run_evals(pairs, cache, cache_f, workers, pess, label):
    todo = [pr for pr in pairs if f"{pr[0]}||{pr[1]}" not in cache]
    if not todo:
        return
    print(f"evaluating {len(todo)} (strategy, opponents) pairs for {label} on {workers} workers", flush=True)
    t0 = time.time()
    with Pool(workers, initializer=init_sim, initargs=(pess,)) as pool:
        for i, (pr, v) in enumerate(pool.imap_unordered(evaluate, todo), 1):
            cache[f"{pr[0]}||{pr[1]}"] = v
            if i % 200 == 0 or i == len(todo):
                json.dump(cache, open(cache_f, "w"))
                print(f"[{(time.time() - t0) / 60:6.1f} min] {i}/{len(todo)}", flush=True)


def neighbourhood_table(cache, cache_f, workers, pess):
    """{cell: largest solved p safe against every neighbouring cell's true opponents}."""
    pairs = set()
    for c, b in G["bank"].items():
        for d in neighbours(c):
            dt = cell_str(d)
            pairs |= {("cfr", dt), ("p0", dt)} | {(f"{cell_str(c)}|{p:g}", dt) for p in b["paths"] if p > 0}
    run_evals(pairs, cache, cache_f, workers, pess, "the neighbourhood table")
    table = {}
    for c, b in G["bank"].items():
        safe = [p for p in sorted(b["paths"]) if p > 0 and all(
            cache[f"{cell_str(c)}|{p:g}||{cell_str(d)}"]["worst"] >= base_worst(cache, cell_str(d)) - TOL
            for d in neighbours(c))]
        table[cell_str(c)] = max(safe, default=0.0)
    return table


# ---------------------------------------------------------------- driver

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    ap.add_argument("--policies", nargs="+", default=POLICIES)
    ap.add_argument("--seeds", type=int, default=20); ap.add_argument("--hands", type=int, default=300)
    ap.add_argument("--every", type=int, default=25); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--pess", action="store_true", help="p* from results_het_pess (pessimistic punisher)")
    ap.add_argument("--forget", type=float, default=1.0, help="identifier forgetting per hand (1 = none)")
    ap.add_argument("--switch", type=float, default=0.0, help="identifier fixed-share rate (0 = none)")
    ap.add_argument("--time-one", action="store_true"); ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    global OUT
    OUT = "adapt_pess" if a.pess else "adapt"       # p* differs, so simulations must not be shared
    if a.forget < 1.0:
        OUT += f"_forget{a.forget:g}"
    if a.switch > 0.0:
        OUT += f"_switch{a.switch:g}"
    os.makedirs(OUT, exist_ok=True)
    sim_file = lambda j: os.path.join(OUT, f"sim_{j[0]}_s{j[1]}_{j[2]}.json")
    jobs = [(sc, sd, po, a.hands, a.every, a.forget, a.switch) for sc in a.scenarios for sd in range(a.seeds)
            for po in a.policies]
    init_sim(a.pess)
    # Evaluations (exact EV, pessimistic worst case of a given strategy against given opponents) do
    # not depend on how p* was chosen, so one cache serves both runs; it starts from cross_eval.py's.
    cache_f = "adapt_evals.json"
    cache = json.load(open(cache_f)) if os.path.exists(cache_f) else {}
    xf = os.path.join("cross_eval", "evals_s0.json")
    if os.path.exists(xf):
        cache = {**json.load(open(xf)), **cache}
    nbhd_f = "adapt_nbhd.json"                      # also independent of p*
    nbhd = {}
    if any(p.startswith("nbhd") for p in a.policies):
        if os.path.exists(nbhd_f):
            nbhd = json.load(open(nbhd_f))
        elif not a.report:
            nbhd = neighbourhood_table(cache, cache_f, a.workers, a.pess)
            json.dump(nbhd, open(nbhd_f, "w"), indent=1)
            print(f"neighbourhood-safe p for {len(nbhd)} cells; {sum(v > 0 for v in nbhd.values())} above 0", flush=True)
    G["nbhd"] = nbhd
    matrix = build_matrix(cache) if any(p.startswith("sel") for p in a.policies) else {}
    G["matrix"] = matrix
    if matrix:
        print(f"bank matrix: {len(matrix)} opponent cells scored, ~{int(np.median([len(v) for v in matrix.values()]))} "
              f"strategies each", flush=True)
    if a.time_one:
        t0 = time.time()
        r = simulate((a.scenarios[0], 0, a.policies[0], a.hands, a.every))
        t1 = time.time()
        pairs = {(k, t) for _, _, k, t in segments(r["scenario"], r["log"], a.hands)}
        for pr in pairs:
            evaluate(pr)
        t2 = time.time()
        print(f"one session ({a.hands} hands): {t1 - t0:.0f}s; its {len(pairs)} distinct evaluations: {t2 - t1:.0f}s")
        return
    if not a.report:
        todo = [j for j in jobs if not os.path.exists(sim_file(j))]
        print(f"simulating {len(todo)} of {len(jobs)} sessions on {a.workers} workers", flush=True)
        t0 = time.time()
        with Pool(a.workers, initializer=init_sim, initargs=(a.pess, nbhd, matrix)) as pool:
            for i, r in enumerate(pool.imap_unordered(simulate, todo), 1):
                json.dump(r, open(sim_file((r["scenario"], r["seed"], r["policy"])), "w"))
                if i % 40 == 0:
                    print(f"[{(time.time() - t0) / 60:6.1f} min] {i}/{len(todo)} sessions", flush=True)
    sims = [json.load(open(sim_file(j))) for j in jobs if os.path.exists(sim_file(j))]
    need = set()
    for r in sims:
        need |= {(k, t) for _, _, k, t in segments(r["scenario"], r["log"], r["hands"])}
    for sc in a.scenarios:
        for t in truths_of(sc):
            need |= {(k, t) for k in candidates()}           # baselines, cell oracle and hindsight oracle
    if not a.report:
        run_evals(need, cache, cache_f, a.workers, a.pess, "sessions and oracles")
    report(a, sims, cache)


def report(a, sims, cache):
    E = lambda k, t: cache.get(f"{k}||{t}")
    def score(scen, log, hands):
        segs = segments(scen, log, hands)
        if any(E(k, t) is None for _, _, k, t in segs) or any(E(b, t) is None for b in ("cfr", "p0") for _, _, _, t in segs):
            return None
        ev = sum((hi - lo) * E(k, t)["ev"] for lo, hi, k, t in segs) / hands
        bw = {t: base_worst(cache, t) for _, _, _, t in segs}
        exposed = sum((hi - lo) for lo, hi, k, t in segs if E(k, t)["worst"] < bw[t] - TOL) / hands
        worst = min(E(k, t)["worst"] - bw[t] for _, _, k, t in segs)
        return dict(ev=ev, exposed=exposed, worst=worst)
    print(f"\n{'scenario':<17}{'policy':<12}{'n':>4}{'avg EV':>9}{'shortfall':>11}{'exposed':>9}"
          f"{'worst vs base':>15}{'commit':>8}")
    for sc in a.scenarios:
        ref = {w: score(sc, reference_log(sc, w, a.hands, a.every, cache), a.hands) for w in ("p0", "cell", "best")}
        if ref["best"] is None:
            continue
        top = ref["best"]["ev"] * a.hands
        for w in ("p0", "cell", "best"):
            r = ref[w]
            if r:
                print(f"{sc:<17}{w + ' (ref)':<12}{'-':>4}{r['ev']:>+9.3f}{top - r['ev'] * a.hands:>+11.1f}"
                      f"{r['exposed']:>9.0%}{r['worst']:>+15.3f}{'-':>8}")
        for po in a.policies:
            S = [s for s in sims if s["scenario"] == sc and s["policy"] == po]
            R = [x for x in (score(sc, s["log"], s["hands"]) for s in S) if x]
            if not R:
                continue
            commit = [next((e["hand"] for e in s["log"] if e["p"] > 0), a.hands) for s in S]
            print(f"{sc:<17}{po:<12}{len(R):>4}{np.median([r['ev'] for r in R]):>+9.3f}"
                  f"{np.median([top - r['ev'] * a.hands for r in R]):>+11.1f}"
                  f"{np.median([r['exposed'] for r in R]):>9.0%}{np.median([r['worst'] for r in R]):>+15.3f}"
                  f"{np.median(commit):>8.0f}")
    print("\navg EV: per hand vs equilibrium, exact. shortfall: chips behind the hindsight-best safe bank "
          "strategy over the session (median). exposed: share of hands whose strategy's pessimistic worst "
          "case is more than 0.05 below the best equilibrium-level strategy's. worst vs base: the session's "
          "lowest such margin. commit: first hand with p > 0. References: p0 = never adapt; cell = the true "
          "opponents' nearest cell at its p*; best = hindsight.")


if __name__ == "__main__":
    main()
