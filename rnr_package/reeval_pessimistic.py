"""
Re-evaluate every result with a PESSIMISTIC punisher, so the worst case cannot depend on
best-response tie-breaking.

The punisher still maximises its own payoff, but among its actions within EPS chips of its
best (per information set, conditional on reaching it) it picks the one worst for us. The
audit showed this matters a lot when everyone else is at equilibrium and little when a biased
opponent is at the table; making it the default removes the question entirely.

For every setting (pair, alpha, seat) that has a linear baseline, writes
results/pess/pess_<pair>_a<alpha>_s<seat>.json with, for each strategy:
    gain       EV against the biased pair, relative to the seat's equilibrium value
    worst      pessimistic worst case over both punishers, relative to equilibrium
Strategies: CFR (5k), CFR (1k, if --cfr1k given), linear mixes w = 0..1, every saved RNR
strategy. Resumable; shardable across processes.

    python reeval_pessimistic.py                       # everything, one process
    python reeval_pessimistic.py --shard 0 --nshards 8 # one of 8 parallel workers
"""

import argparse
import glob
import json
import os
import pickle
import re
import time

import numpy as np

from tree_engine import Tree
from q1_alpha_sweep import perturb

RESULTS = "results"
OUTDIR = os.path.join(RESULTS, "pess")


def respond_pess(T, sig, player, score_u, track_u, eps):
    pe = T.edge_probs(sig, skip=player); rmi = T.reach(pe)
    vs, vt = score_u.copy(), track_u.copy()
    nSA, off, sai = T.nSA[player], T.off[player], T.sai[player]
    chosen = np.zeros(nSA, dtype=bool)
    for d in range(T.maxd, 0, -1):
        nd = T.by_depth[d]; mine = nd[T.kind[nd] == player]
        if len(mine):
            w = rmi[T.parent[mine]]
            W = np.bincount(T.esa[mine], weights=w, minlength=nSA)
            sc = np.bincount(T.esa[mine], weights=w * vs[mine], minlength=nSA)
            tr = np.bincount(T.esa[mine], weights=w * vt[mine], minlength=nSA)
            touched = np.zeros(nSA, dtype=bool); touched[T.esa[mine]] = True
            ok = touched & (W > 0)
            scc = np.where(ok, sc / np.where(W > 0, W, 1), -np.inf)
            trc = np.where(ok, tr / np.where(W > 0, W, 1), np.inf)
            # information sets reached with zero probability: fall back to the plain best
            scz = np.where(touched & (W == 0), sc, -np.inf)
            best = np.maximum.reduceat(np.where(ok, scc, scz), off[:-1])
            cand = touched & (np.where(ok, scc, scz) >= best[sai] - np.where(ok, eps, 1e-12)[...])
            key = np.where(cand, np.where(ok, trc, 0.0), np.inf)
            lo = np.minimum.reduceat(key, off[:-1])
            pick = cand & (key <= lo[sai])
            idx = np.where(pick, np.arange(nSA), nSA)
            first = np.minimum.reduceat(idx, off[:-1]); good = first < nSA
            chosen[first[good]] = True
            pe[mine] = chosen[T.esa[mine]].astype(float)
        vs += np.bincount(T.parent[nd], weights=pe[nd] * vs[nd], minlength=T.N)
        vt += np.bincount(T.parent[nd], weights=pe[nd] * vt[nd], minlength=T.N)
    return vt[0]


def load_table(path):
    blob = pickle.load(open(path, "rb"))
    return blob["table"], blob.get("iters")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfr", default=None, help="5k table (default: cfr_3p_5000.pkl, else cfr_3p_1000.pkl)")
    ap.add_argument("--cfr1k", default=None, help="optional 1,000-iteration table, added as a baseline")
    ap.add_argument("--eps", type=float, default=0.01)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()

    cfr = a.cfr or next(f for f in ("cfr_3p_5000.pkl", "cfr_3p_1000.pkl") if os.path.exists(f))
    base, iters = load_table(cfr)
    print(f"baseline {cfr} ({iters} iterations), eps {a.eps}", flush=True)
    b1 = load_table(a.cfr1k)[0] if a.cfr1k else None

    T = Tree(); U = T.util
    B5 = [T.from_table(base, q) for q in range(3)]
    B1 = [T.from_table(b1, q) for q in range(3)] if b1 else None
    EQ = T.ev(B5)

    settings = sorted({(r["pair"], r["alpha"], r["seat"]) for r in
                       (json.load(open(f)) for f in glob.glob(os.path.join(RESULTS, "linear_*.json")))})
    mine = [k for i, k in enumerate(settings) if i % a.nshards == a.shard]
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"{len(mine)} of {len(settings)} settings in this shard", flush=True)

    for pair, alpha, s in mine:
        out = os.path.join(OUTDIR, f"pess_{pair}_a{alpha}_s{s}.json")
        if os.path.exists(out):
            continue
        t0 = time.time()
        opps = [o for o in range(3) if o != s]
        fix = {o: T.from_table(perturb(base, t, alpha), o) for o, t in zip(opps, pair)}
        e = EQ[s]

        def evaluate(strat):
            prof = [None] * 3; prof[s] = strat
            for o in opps:
                prof[o] = fix[o]
            gain = T.ev(prof)[s] - e
            worst = []
            for o in opps:
                other = [x for x in opps if x != o][0]
                pr = [None] * 3; pr[s], pr[other] = strat, fix[other]
                worst.append(respond_pess(T, pr, o, U[:, o], U[:, s], a.eps) - e)
            return {"gain": gain, "worst": min(worst), "worst_by_punisher": dict(zip(map(str, opps), worst))}

        res = {"pair": pair, "alpha": alpha, "seat": s, "eps": a.eps, "baseline_iters": iters}
        res["cfr5k"] = evaluate(B5[s])
        if B1:
            res["cfr1k"] = evaluate(B1[s])
        prof = [None] * 3
        for o in opps:
            prof[o] = fix[o]
        br, _ = T.respond(prof, s, U[:, s], U[:, s])
        res["linear"] = {f"{w:.1f}": evaluate((1 - w) * B5[s] + w * br) for w in np.linspace(0, 1, 11)}
        res["rnr"] = {}
        pat = re.compile(rf"rnr_{pair}_a{alpha}_s{s}_independent_selfish_p([0-9.]+)_strategy\.npy$")
        for f in glob.glob(os.path.join(RESULTS, f"rnr_{pair}_a{alpha}_s{s}_independent_selfish_p*_strategy.npy")):
            m = pat.search(os.path.basename(f))
            if m:                                   # skip suffixed re-solves
                res["rnr"][m.group(1)] = evaluate(np.load(f))
        json.dump(res, open(out, "w"), indent=1)
        print(f"{pair} a{alpha} s{s}: {len(res['rnr'])} RNR strategies, {time.time()-t0:.0f}s", flush=True)
    open(os.path.join(OUTDIR, f"_shard{a.shard}of{a.nshards}.done"), "w").write("done")


if __name__ == "__main__":
    main()
