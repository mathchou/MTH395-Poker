"""
General three-player restricted Nash response (RNR).

Covers RERUN-21 (symmetric restriction), RERUN-22 (self-interested free opponents),
RERUN-10 (other pairs and alphas) and RERUN-16 (other seats) with one solver.

Modified game: the learner sits in --seat; the other two seats get opponent types from
--pair (in increasing seat order). Each opponent is either FIXED (plays its biased
model) or FREE (knows it is free, optimises). The learner never observes the modes.

  --restrict one-sided     first opponent free w.p. 1-p, second always fixed
                           (reproduces rnr3p.py / draft v5 Section 6)
  --restrict independent   each opponent independently free w.p. 1-p   [RERUN-21]

  --free adversarial       a free opponent minimises the learner's payoff
  --free selfish           a free opponent maximises its own payoff      [RERUN-22]

independent + adversarial lets BOTH opponents minimise our payoff in the same hand,
i.e. a coalition. That violates the no-collusion assumption and is refused.

Output: results/rnr_<tag>.json with the learner's strategy summary, convergence gaps,
and evaluation against the biased pair and against single punishers. A matching
linear-mixing baseline is written to results/linear_<tag>.json.

    python rnr_general.py --pair FF --alpha 0.35 --seat 0 --p 0.5 \
        --restrict independent --free selfish --iters 400
"""

import argparse
import itertools
import json
import os
import pickle
import time

import numpy as np

from tree_engine import Tree
from q1_alpha_sweep import perturb

OUT = "results"


def load(pair, alpha, seat, cfr="cfr_3p_1000.pkl"):
    T = Tree()
    base = pickle.load(open(cfr, "rb"))["table"]
    opps = [o for o in range(3) if o != seat]
    fix = {o: T.from_table(perturb(base, t, alpha), o) for o, t in zip(opps, pair)}
    base_sig = [T.from_table(base, q) for q in range(3)]
    return T, base_sig, opps, fix


def own_reach_sa(T, sig_q, q):
    pe = np.ones(T.N)
    idx = T.kmask[q]
    pe[idx] = sig_q[T.esa[idx]]
    r = T.reach(pe)
    out = np.zeros(T.nSA[q]); out[T.esa[idx]] = r[T.parent[idx]]
    return out


def mixture(T, q, fixed, free, pfix):
    """Behaviour strategy equivalent to 'fixed w.p. pfix, else free' (reach-weighted)."""
    if pfix >= 1:
        return fixed
    rf, rr = own_reach_sa(T, fixed, q), own_reach_sa(T, free, q)
    num = pfix * rf * fixed + (1 - pfix) * rr * free
    den = pfix * rf + (1 - pfix) * rr
    return np.where(den > 0, num / np.where(den > 0, den, 1), fixed)


def combos(opps, p, restrict):
    """List of (probability, {opp: 'fix'|'free'})."""
    if restrict == "one-sided":
        a, b = opps
        return [(p, {a: "fix", b: "fix"}), (1 - p, {a: "free", b: "fix"})]
    out = []
    for ma, mb in itertools.product(["fix", "free"], repeat=2):
        pr = (p if ma == "fix" else 1 - p) * (p if mb == "fix" else 1 - p)
        out.append((pr, {opps[0]: ma, opps[1]: mb}))
    return out


def solve(args):
    if args.restrict == "independent" and args.free == "adversarial":
        raise SystemExit("independent + adversarial = coalition of free opponents; "
                         "violates no-collusion. Use --free selfish.")
    T, base_sig, opps, fix = load(args.pair, args.alpha, args.seat)
    s = args.seat
    U = T.util
    tag = f"{args.pair}_a{args.alpha}_s{s}_{args.restrict}_{args.free}_p{args.p}"
    if args.suffix:
        tag += f"_{args.suffix}"
    use_cache = not args.suffix and args.seed is None      # independent re-solves bypass the cache
    os.makedirs(OUT, exist_ok=True)

    # At p = 0 with independent restriction both opponents are always free, so the
    # biased models never enter the modified game: the solve is identical for every
    # pair and every alpha. Solve once per (seat, free) and reuse; only the evaluation
    # depends on the pair.
    cache = f"{OUT}/p0cache_s{s}_{args.free}.npz"
    if use_cache and args.restrict == "independent" and args.p == 0 and os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        s0, gaps = z["s0"], z["gaps"].item()
        print(f"  p=0 solve reused from {cache}", flush=True)
        res = {"pair": args.pair, "alpha": args.alpha, "seat": s, "p": args.p,
               "restrict": args.restrict, "free": args.free, "iters": int(z["iters"]),
               "gaps": gaps, "p0_cached": True, **evaluate(T, s0, s, opps, fix)}
        json.dump(res, open(f"{OUT}/rnr_{tag}.json", "w"), indent=1)
        np.save(f"{OUT}/rnr_{tag}_strategy.npy", s0)
        print(json.dumps({k: v for k, v in res.items() if k != "gaps"}, indent=1))
        return
    util_of = {o: (-U[:, s] if args.free == "adversarial" else U[:, o]) for o in opps}
    cmb = [c for c in combos(opps, args.p, args.restrict) if c[0] > 0]
    free_opps = sorted({o for _, m in cmb for o in opps if m[o] == "free"})

    sig = {s: T.uniform(s), **{o: T.uniform(o) for o in free_opps}}
    if args.seed is not None:
        # random FIRST-iteration strategies; regrets still start at zero (seeding regrets
        # with a strategy acts as a prior worth thousands of iterations)
        rng = np.random.default_rng(args.seed)
        sig = {q: T.normalize(rng.random(T.nSA[q]) + 1e-3, q) for q in sig}
    R = {q: np.zeros(T.nSA[q]) for q in sig}
    W = {q: np.zeros(T.nSA[q]) for q in sig}

    def profile(modes, cur):
        prof = [None] * 3
        prof[s] = cur[s]
        for o in opps:
            prof[o] = fix[o] if modes[o] == "fix" else cur[o]
        return prof

    t0 = time.time()
    for t in range(1, args.iters + 1):
        # learner
        c = np.zeros(T.nSA[s])
        for pr, m in cmb:
            c += pr * T.cfv(profile(m, sig), s, U[:, s])
        ev = np.bincount(T.sai[s], weights=sig[s] * c, minlength=T.nI[s])[T.sai[s]]
        R[s] = np.maximum(R[s] + c - ev, 0)
        sig[s] = T.rm(R[s], s)
        W[s] += t * own_reach_sa(T, sig[s], s) * sig[s]
        # free opponents
        for o in free_opps:
            c = np.zeros(T.nSA[o])
            for pr, m in cmb:
                if m[o] == "free":
                    c += pr * T.cfv(profile(m, sig), o, util_of[o])
            ev = np.bincount(T.sai[o], weights=sig[o] * c, minlength=T.nI[o])[T.sai[o]]
            R[o] = np.maximum(R[o] + c - ev, 0)
            sig[o] = T.rm(R[o], o)
            W[o] += t * own_reach_sa(T, sig[o], o) * sig[o]
        if t % 50 == 0 or t == args.iters:
            print(f"  iter {t}/{args.iters}  [{time.time()-t0:.0f}s]", flush=True)

    avg = {q: T.normalize(W[q], q) for q in sig}

    # ---- convergence: each role's best-response gain in the modified game
    pfix_of = {o: (1.0 if o not in free_opps else
                   sum(pr for pr, m in cmb if m[o] == "fix")) for o in opps}
    mix = {o: mixture(T, o, fix[o], avg.get(o, fix[o]), pfix_of[o]) for o in opps}
    obj = sum(pr * T.ev(profile(m, avg))[s] for pr, m in cmb)
    prof_mix = [None] * 3
    for o in opps:
        prof_mix[o] = mix[o]
    _, br_val = T.respond(prof_mix, s, U[:, s], U[:, s])
    gaps = {"learner": br_val - obj}
    for o in free_opps:
        other = [x for x in opps if x != o][0]
        prof = [None] * 3
        prof[s], prof[other] = avg[s], mix[other]
        _, v_br = T.respond(prof, o, util_of[o], util_of[o])
        prof[o] = avg[o]
        pe = T.edge_probs(prof)
        v_now = T.values(pe, util_of[o])[0]
        gaps[f"free_{o}"] = v_br - v_now

    res = {"pair": args.pair, "alpha": args.alpha, "seat": s, "p": args.p,
           "restrict": args.restrict, "free": args.free, "iters": args.iters,
           "gaps": gaps, **evaluate(T, avg[s], s, opps, fix)}
    if use_cache and args.restrict == "independent" and args.p == 0:
        np.savez(cache, s0=avg[s], gaps=np.array(gaps, dtype=object), iters=args.iters)
    json.dump(res, open(f"{OUT}/rnr_{tag}.json", "w"), indent=1)
    np.save(f"{OUT}/rnr_{tag}_strategy.npy", avg[s])
    print(json.dumps({k: v for k, v in res.items() if k != "gaps"}, indent=1))
    print("gaps:", {k: round(v, 5) for k, v in gaps.items()})


def evaluate(T, sig_s, s, opps, fix):
    """EV vs biased pair, and vs each single punisher (other opponent stays biased)."""
    U = T.util
    prof = [None] * 3; prof[s] = sig_s
    for o in opps:
        prof[o] = fix[o]
    out = {"vs_biased": T.ev(prof)[s]}
    for o in opps:
        other = [x for x in opps if x != o][0]
        pr = [None] * 3; pr[s], pr[other] = sig_s, fix[other]
        _, adv = T.respond(pr, o, -U[:, s], U[:, s])
        _, slf = T.respond(pr, o, U[:, o], U[:, s])
        out[f"adversary_seat{o}"], out[f"selfish_seat{o}"] = adv, slf
    out["worst_adversary"] = min(out[f"adversary_seat{o}"] for o in opps)
    out["worst_selfish"] = min(out[f"selfish_seat{o}"] for o in opps)
    return out


def linear(args):
    T, base_sig, opps, fix = load(args.pair, args.alpha, args.seat)
    s = args.seat
    prof = [None] * 3
    for o in opps:
        prof[o] = fix[o]
    br, _ = T.respond(prof, s, T.util[:, s], T.util[:, s])
    rows = []
    for w in np.round(np.linspace(0, 1, 11), 2):
        e = evaluate(T, (1 - w) * base_sig[s] + w * br, s, opps, fix)
        rows.append({"w": float(w), **e})
        print(f"  w={w:.1f}  vs_biased {e['vs_biased']:+.3f}  worst_adv "
              f"{e['worst_adversary']:+.3f}  worst_selfish {e['worst_selfish']:+.3f}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump({"pair": args.pair, "alpha": args.alpha, "seat": s, "rows": rows},
              open(f"{OUT}/linear_{args.pair}_a{args.alpha}_s{s}.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="FF")
    ap.add_argument("--alpha", type=float, default=0.35)
    ap.add_argument("--seat", type=int, default=0)
    ap.add_argument("--p", type=float, default=0.5)
    ap.add_argument("--restrict", choices=["one-sided", "independent"], default="independent")
    ap.add_argument("--free", choices=["adversarial", "selfish"], default="selfish")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--suffix", default="", help="appended to output names, for independent re-solves")
    ap.add_argument("--seed", type=int, default=None, help="random first-iteration strategies")
    ap.add_argument("--linear", action="store_true", help="linear-mixing baseline only")
    a = ap.parse_args()
    linear(a) if a.linear else solve(a)
