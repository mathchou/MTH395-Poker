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

    python rnr_general.py --pair FF --alpha 0.35 --seat 0 --p 0.5 \\
        --restrict independent --free selfish --iters 400

Different bias for each opponent: --alpha applies to the first opponent, --alpha2 to the
second. Mixed-alpha runs should go to their own folder, so existing analysis scripts (which
group results by a single alpha) never mix them in:

    python rnr_general.py --pair FR --alpha 0.35 --alpha2 0.1 --out results_het

HOW THE SOLVER WORKS (read this first)
--------------------------------------
OpenSpiel's CFR+ solves a game in which every player learns. A restricted Nash response
needs a *modified* game: before each hand, chance decides for each opponent whether it is
FIXED (forced to play its biased model) or FREE (plays whatever it likes, knowing it is
free). We never see these decisions.

This file never builds that modified game as a tree. Instead it notes that the modified
game is just a small number of "worlds" — one per combination of fixed/free — each of
which is the ordinary Leduc tree with particular strategies plugged into each seat. So
every CFR+ iteration:

    for each world (with its probability):
        plug in: our current strategy; for each opponent, its biased model if fixed
                 in this world, or its current learned strategy if free
        compute counterfactual values on the ordinary tree (tree_engine.cfv)
    add them up, weighted by world probability
    do the usual CFR+ step (regrets, floor at zero, regret matching, averaging)

Our strategy is the SAME in every world, because we cannot tell which world we are in.
A free opponent's strategy is the same in every world where it is free, because it
cannot see the other opponent's mode. Fixed opponents never learn. Mathematically this
is identical to running CFR+ on the modified game: the world probabilities play the role
of its extra chance node, and sharing strategies across worlds plays the role of its
information sets.

Vocabulary used in the comments:
    learner        us, the seat whose restricted response we compute (args.seat, `s`)
    world / mode   one combination of fixed/free for the two opponents
    SA pair        an (information set, action) pair; strategies are flat arrays of them
    sig            current strategy (one flat SA array per seat)
    R              cumulative regrets;  W  accumulated average-strategy weights
"""

import argparse
import itertools
import json
import os
import pickle
import time

import numpy as np

from tree_engine import Tree            # the flattened game tree and its passes
from q1_alpha_sweep import perturb      # builds a biased opponent from the equilibrium
from cfr_path import find_cfr           # which CFR table to read (shared by all scripts)

OUT = "results"                         # every output file goes here (--out)
CACHE_DIR = "results"                   # where the shared p = 0 solve is cached (--cache-dir)


# ---------------------------------------------------------------------------------
# Setup: the game tree, the equilibrium, and the two biased opponents
# ---------------------------------------------------------------------------------

def load(pair, alpha, seat, cfr=None, alpha2=None):
    """
    Returns
        T         the tree engine (1.83M-node Leduc tree as arrays)
        base_sig  the CFR equilibrium, one flat strategy array per seat
        opps      the two opponent seats, in increasing order (e.g. [1, 2] when seat = 0)
        fix       {opponent seat: its biased strategy}; pair letter k goes to opps[k]

    alpha is the first opponent's bias; alpha2 the second's (None = same as alpha).
    """
    T = Tree()
    base = pickle.load(open(find_cfr(cfr), "rb"))["table"]          # the equilibrium as a dict
    opps = [o for o in range(3) if o != seat]                        # everyone except us
    # Pair "FR" from seat 0: opps = [1, 2], so seat 1 gets type F and seat 2 type R.
    # perturb() tilts the equilibrium toward the favoured action by alpha.
    alphas = [alpha, alpha if alpha2 is None else alpha2]           # one bias per opponent
    fix = {o: T.from_table(perturb(base, t, a), o) for o, t, a in zip(opps, pair, alphas)}
    base_sig = [T.from_table(base, q) for q in range(3)]            # equilibrium, flat form
    return T, base_sig, opps, fix


# ---------------------------------------------------------------------------------
# Helpers for averaging strategies correctly
# ---------------------------------------------------------------------------------

def own_reach_sa(T, sig_q, q):
    """
    For seat q playing sig_q: the probability that q's OWN choices lead to each of its
    information sets, returned per SA pair (all actions of an infoset get the same value).

    Why it's needed: an average strategy must weight each iteration's strategy at an
    information set by how often the player actually plays into it. A strategy that
    almost never reaches an information set shouldn't dominate the average there.
    """
    pe = np.ones(T.N)                          # every move counts as certain ...
    idx = T.kmask[q]                           # ... except seat q's own moves,
    pe[idx] = sig_q[T.esa[idx]]                # which get q's probabilities
    r = T.reach(pe)                            # forward pass: q's own reach of every node
    # Each SA pair takes the reach of the decision node it leaves from (its parent).
    # All nodes in an information set share q's own reach (perfect recall), so writing
    # the same value repeatedly is harmless.
    out = np.zeros(T.nSA[q]); out[T.esa[idx]] = r[T.parent[idx]]
    return out


def mixture(T, q, fixed, free, pfix):
    """
    Behaviour strategy equivalent to 'fixed w.p. pfix, else free' (reach-weighted).

    Seat q follows `fixed` with probability pfix and `free` otherwise, decided once
    before the hand. A single strategy that produces the same play must, at each
    information set, weight the two by how likely each version is to have arrived
    there: pfix * (fixed version's reach) vs (1 - pfix) * (free version's reach).
    Used only for the convergence check below.
    """
    if pfix >= 1:                               # always fixed: nothing to mix
        return fixed
    rf, rr = own_reach_sa(T, fixed, q), own_reach_sa(T, free, q)
    num = pfix * rf * fixed + (1 - pfix) * rr * free
    den = pfix * rf + (1 - pfix) * rr
    # Where neither version can reach an information set, its strategy there is
    # irrelevant; fall back to the fixed one to avoid dividing by zero.
    return np.where(den > 0, num / np.where(den > 0, den, 1), fixed)


# ---------------------------------------------------------------------------------
# The worlds of the modified game
# ---------------------------------------------------------------------------------

def combos(opps, p, restrict):
    """
    List of (probability, {opp: 'fix'|'free'}).

    p is the probability that an opponent follows its model ('fix').
      one-sided:    only the first opponent can be free; the second is always fixed
                    -> 2 worlds: (p, both fixed), (1 - p, first free)
      independent:  each opponent independently fixed w.p. p
                    -> 4 worlds; at p = 0.5 each has probability 0.25
    """
    if restrict == "one-sided":
        a, b = opps
        return [(p, {a: "fix", b: "fix"}), (1 - p, {a: "free", b: "fix"})]
    out = []
    for ma, mb in itertools.product(["fix", "free"], repeat=2):          # 4 combinations
        pr = (p if ma == "fix" else 1 - p) * (p if mb == "fix" else 1 - p)   # independent
        out.append((pr, {opps[0]: ma, opps[1]: mb}))
    return out


# ---------------------------------------------------------------------------------
# The solver
# ---------------------------------------------------------------------------------

def alpha_tag(args):
    """'0.35' when both opponents share alpha (unchanged file names); '0.35-0.1' when not."""
    if args.alpha2 is None or args.alpha2 == args.alpha:
        return f"{args.alpha}"
    return f"{args.alpha}-{args.alpha2}"


def solve(args):
    # Two adversarial free opponents in the same world would both be minimising OUR
    # payoff together, i.e. colluding. The study assumes no collusion, so refuse.
    if args.restrict == "independent" and args.free == "adversarial":
        raise SystemExit("independent + adversarial = coalition of free opponents; "
                         "violates no-collusion. Use --free selfish.")
    T, base_sig, opps, fix = load(args.pair, args.alpha, args.seat, alpha2=args.alpha2)
    s = args.seat                             # our seat
    U = T.util                                # payoffs at every node: U[:, seat]
    # Output file name, e.g. FF_a0.35_s0_independent_selfish_p0.5[_suffix]
    tag = f"{args.pair}_a{alpha_tag(args)}_s{s}_{args.restrict}_{args.free}_p{args.p}"
    if args.suffix:
        tag += f"_{args.suffix}"
    use_cache = not args.suffix and args.seed is None      # independent re-solves bypass the cache
    os.makedirs(OUT, exist_ok=True)

    # At p = 0 with independent restriction both opponents are always free, so the
    # biased models never enter the modified game: the solve is identical for every
    # pair and every alpha. Solve once per (seat, free) and reuse; only the evaluation
    # depends on the pair.
    # (Also independent of the opponents' alphas, so mixed-alpha runs can share it.)
    cache = f"{CACHE_DIR}/p0cache_s{s}_{args.free}.npz"
    if use_cache and args.restrict == "independent" and args.p == 0 and os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        s0, gaps = z["s0"], z["gaps"].item()          # our strategy and its convergence gaps
        print(f"  p=0 solve reused from {cache}", flush=True)
        res = {"pair": args.pair, "alpha": args.alpha, "alpha2": args.alpha if args.alpha2 is None else args.alpha2, "seat": s, "p": args.p,
               "restrict": args.restrict, "free": args.free, "iters": int(z["iters"]),
               "gaps": gaps, "p0_cached": True, **evaluate(T, s0, s, opps, fix)}
        json.dump(res, open(f"{OUT}/rnr_{tag}.json", "w"), indent=1)
        np.save(f"{OUT}/rnr_{tag}_strategy.npy", s0)
        print(json.dumps({k: v for k, v in res.items() if k != "gaps"}, indent=1))
        return

    # What a free opponent maximises: our payoff with a minus sign (adversarial) or its
    # own payoff (selfish). This is the only difference between the two settings.
    util_of = {o: (-U[:, s] if args.free == "adversarial" else U[:, o]) for o in opps}
    # The worlds, dropping any with probability 0 (e.g. at p = 1, every world with a
    # free opponent disappears and the solve becomes a plain best response).
    cmb = [c for c in combos(opps, args.p, args.restrict) if c[0] > 0]
    # Opponents that are free in at least one world: these are the only ones that learn.
    free_opps = sorted({o for _, m in cmb for o in opps if m[o] == "free"})

    # ---- starting strategies: uniform for every learner ----
    sig = {s: T.uniform(s), **{o: T.uniform(o) for o in free_opps}}
    if args.seed is not None:
        # random FIRST-iteration strategies; regrets still start at zero (seeding regrets
        # with a strategy acts as a prior worth thousands of iterations)
        rng = np.random.default_rng(args.seed)
        sig = {q: T.normalize(rng.random(T.nSA[q]) + 1e-3, q) for q in sig}
    R = {q: np.zeros(T.nSA[q]) for q in sig}       # cumulative regrets, per learner
    W = {q: np.zeros(T.nSA[q]) for q in sig}       # average-strategy accumulators

    def profile(modes, cur):
        """
        The strategies actually at the table in one world: ours (current), and for each
        opponent its biased model if fixed in this world, else its current learned strategy.
        """
        prof = [None] * 3
        prof[s] = cur[s]
        for o in opps:
            prof[o] = fix[o] if modes[o] == "fix" else cur[o]
        return prof

    t0 = time.time()
    for t in range(1, args.iters + 1):
        # ---- our update ------------------------------------------------------------
        # Counterfactual value of each of our SA pairs, summed over worlds weighted by
        # their probability: we optimise against the mixture of worlds, since we can't
        # tell them apart.
        c = np.zeros(T.nSA[s])
        for pr, m in cmb:
            c += pr * T.cfv(profile(m, sig), s, U[:, s])
        # Value of each information set under our current strategy (sum over its actions
        # of probability x counterfactual value), copied back onto every SA pair of it.
        ev = np.bincount(T.sai[s], weights=sig[s] * c, minlength=T.nI[s])[T.sai[s]]
        # CFR+: add this iteration's regrets (c - ev) and floor at zero.
        R[s] = np.maximum(R[s] + c - ev, 0)
        # Regret matching: next strategy proportional to positive regret.
        sig[s] = T.rm(R[s], s)
        # Linear averaging: iteration t gets weight t, times our own reach (see own_reach_sa).
        W[s] += t * own_reach_sa(T, sig[s], s) * sig[s]

        # ---- free opponents' updates -------------------------------------------------
        # Alternating updates: they respond to OUR strategy as just updated above.
        for o in free_opps:
            c = np.zeros(T.nSA[o])
            for pr, m in cmb:
                if m[o] == "free":                 # only the worlds where o is free
                    # pr includes o's own probability of being free, (1 - p): a constant
                    # factor that regret matching ignores (only ratios matter).
                    c += pr * T.cfv(profile(m, sig), o, util_of[o])
            ev = np.bincount(T.sai[o], weights=sig[o] * c, minlength=T.nI[o])[T.sai[o]]
            R[o] = np.maximum(R[o] + c - ev, 0)
            sig[o] = T.rm(R[o], o)
            W[o] += t * own_reach_sa(T, sig[o], o) * sig[o]
        if t % 50 == 0 or t == args.iters:
            print(f"  iter {t}/{args.iters}  [{time.time()-t0:.0f}s]", flush=True)

    # The answer is the AVERAGE strategy, not the last one (as always in CFR).
    avg = {q: T.normalize(W[q], q) for q in sig}

    # ---- convergence: each role's best-response gain in the modified game
    # How much could each learner still gain by switching to a best response, while the
    # others keep their average strategies? Small gaps = close to a solution.
    #
    # To compute OUR best response we need a single strategy for each opponent that
    # behaves like "fixed with probability pfix, otherwise its learned strategy".
    pfix_of = {o: (1.0 if o not in free_opps else
                   sum(pr for pr, m in cmb if m[o] == "fix")) for o in opps}
    mix = {o: mixture(T, o, fix[o], avg.get(o, fix[o]), pfix_of[o]) for o in opps}
    # Our value in the modified game: each world's value, weighted by its probability.
    obj = sum(pr * T.ev(profile(m, avg))[s] for pr, m in cmb)
    prof_mix = [None] * 3
    for o in opps:
        prof_mix[o] = mix[o]
    _, br_val = T.respond(prof_mix, s, U[:, s], U[:, s])      # our best possible value
    gaps = {"learner": br_val - obj}
    # Each free opponent: best response against our average and the other opponent's
    # mixture, compared with what its own average strategy earns there.
    for o in free_opps:
        other = [x for x in opps if x != o][0]
        prof = [None] * 3
        prof[s], prof[other] = avg[s], mix[other]
        _, v_br = T.respond(prof, o, util_of[o], util_of[o])
        prof[o] = avg[o]
        pe = T.edge_probs(prof)
        v_now = T.values(pe, util_of[o])[0]
        gaps[f"free_{o}"] = v_br - v_now

    # ---- evaluate our strategy in the REAL game and save everything ----
    res = {"pair": args.pair, "alpha": args.alpha, "alpha2": args.alpha if args.alpha2 is None else args.alpha2, "seat": s, "p": args.p,
           "restrict": args.restrict, "free": args.free, "iters": args.iters,
           "gaps": gaps, **evaluate(T, avg[s], s, opps, fix)}
    if use_cache and args.restrict == "independent" and args.p == 0:
        np.savez(cache, s0=avg[s], gaps=np.array(gaps, dtype=object), iters=args.iters)
    json.dump(res, open(f"{OUT}/rnr_{tag}.json", "w"), indent=1)
    np.save(f"{OUT}/rnr_{tag}_strategy.npy", avg[s])          # our strategy, flat SA array
    print(json.dumps({k: v for k, v in res.items() if k != "gaps"}, indent=1))
    print("gaps:", {k: round(v, 5) for k, v in gaps.items()})


# ---------------------------------------------------------------------------------
# Evaluation in the real game
# ---------------------------------------------------------------------------------

def evaluate(T, sig_s, s, opps, fix):
    """
    EV vs biased pair, and vs each single punisher (other opponent stays biased).

    Payoffs here are RAW (not relative to our equilibrium value); analysis scripts
    subtract the seat's equilibrium value afterwards.
        vs_biased            our payoff against the two biased opponents
        selfish_seatK        opponent K switches to a best response maximising ITS OWN
                             payoff, the other keeps its model; our payoff
        adversary_seatK      same, but K minimises OUR payoff (measures spite too)
        worst_selfish / worst_adversary   the worse of the two opponents
    Note: these use tree_engine.respond, which breaks exact ties by action order;
    reeval_pessimistic.py gives the tie-breaking-proof version.
    """
    U = T.util
    prof = [None] * 3; prof[s] = sig_s
    for o in opps:
        prof[o] = fix[o]
    out = {"vs_biased": T.ev(prof)[s]}
    for o in opps:
        other = [x for x in opps if x != o][0]
        pr = [None] * 3; pr[s], pr[other] = sig_s, fix[other]   # o's slot left empty: respond fills it
        _, adv = T.respond(pr, o, -U[:, s], U[:, s])            # choose by -our payoff, report ours
        _, slf = T.respond(pr, o, U[:, o], U[:, s])             # choose by its payoff, report ours
        out[f"adversary_seat{o}"], out[f"selfish_seat{o}"] = adv, slf
    out["worst_adversary"] = min(out[f"adversary_seat{o}"] for o in opps)
    out["worst_selfish"] = min(out[f"selfish_seat{o}"] for o in opps)
    return out


def linear(args):
    """
    The linear-mixing baseline: at every information set, play
        (1 - w) * equilibrium + w * best response to the biased pair
    for w = 0, 0.1, ..., 1, and evaluate each. w = 0 is plain CFR, w = 1 the best response.
    """
    T, base_sig, opps, fix = load(args.pair, args.alpha, args.seat, alpha2=args.alpha2)
    s = args.seat
    prof = [None] * 3
    for o in opps:
        prof[o] = fix[o]
    br, _ = T.respond(prof, s, T.util[:, s], T.util[:, s])     # pure best response, flat form
    rows = []
    for w in np.round(np.linspace(0, 1, 11), 2):
        e = evaluate(T, (1 - w) * base_sig[s] + w * br, s, opps, fix)
        rows.append({"w": float(w), **e})
        print(f"  w={w:.1f}  vs_biased {e['vs_biased']:+.3f}  worst_adv "
              f"{e['worst_adversary']:+.3f}  worst_selfish {e['worst_selfish']:+.3f}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump({"pair": args.pair, "alpha": args.alpha, "alpha2": args.alpha if args.alpha2 is None else args.alpha2,
               "seat": s, "rows": rows},
              open(f"{OUT}/linear_{args.pair}_a{alpha_tag(args)}_s{s}.json", "w"), indent=1)


# ---------------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="FF")               # opponent types, in seat order
    ap.add_argument("--alpha", type=float, default=0.35)  # how biased the opponents are (first opponent)
    ap.add_argument("--alpha2", type=float, default=None) # second opponent's bias (default: same as --alpha)
    ap.add_argument("--out", default="results")           # output folder; use results_het for mixed alphas
    ap.add_argument("--cache-dir", default="results")     # where the p = 0 cache lives
    ap.add_argument("--seat", type=int, default=0)        # our seat
    ap.add_argument("--p", type=float, default=0.5)       # probability an opponent follows its model
    ap.add_argument("--restrict", choices=["one-sided", "independent"], default="independent")
    ap.add_argument("--free", choices=["adversarial", "selfish"], default="selfish")
    ap.add_argument("--iters", type=int, default=400)     # CFR+ iterations
    ap.add_argument("--suffix", default="", help="appended to output names, for independent re-solves")
    ap.add_argument("--seed", type=int, default=None, help="random first-iteration strategies")
    ap.add_argument("--linear", action="store_true", help="linear-mixing baseline only")
    a = ap.parse_args()
    OUT, CACHE_DIR = a.out, a.cache_dir                   # module-level settings used by solve/linear
    linear(a) if a.linear else solve(a)
