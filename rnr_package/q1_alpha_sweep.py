"""
Question 1: how does our payout grow with opponent bias alpha?

Opponent types, all derived from the converged CFR table:
    O  optimal (unperturbed CFR)
    R  over-raiser   C  over-caller   F  over-folder
    sigma'(a|I) = (1-alpha) sigma(a|I) + alpha [a == favoured], where legal

For every alpha and every ordered pair (opp in seat 1, opp in seat 2) — 4 x 4 = 16
pairs — we compute, for us in seat 0:
    br      best-response value (the most we can make)
    cfr     what we make by just playing CFR against them
and the FULL return vector in both cases, so we can see where the money comes
from. In three-player poker an opponent's mistake is not automatically our gain:
some of it can flow to the other opponent.

Uses open_spiel's C++ TabularBestResponse, reused across pairs via set_policy
(~5 s per pair). Results are cached; re-running resumes.

    python q1_alpha_sweep.py                 # default alphas
    python q1_alpha_sweep.py --alphas 0.1 0.35 --save-br 0.1 0.35
"""

import argparse
import itertools
import json
import os
import pickle
import time

import numpy as np
import pyspiel
from cfr_path import find_cfr

FAV = {"R": 2, "C": 1, "F": 0}
TYPES = ["O", "R", "C", "F"]
SEAT = 0


def perturb(base, t, alpha):
    if t == "O" or alpha == 0:
        return base
    fav = FAV[t]
    out = {}
    for k, d in base.items():
        if fav not in d:
            out[k] = d
            continue
        n = {a: (1 - alpha) * p for a, p in d.items()}
        n[fav] += alpha
        s = sum(n.values())
        out[k] = {a: p / s for a, p in n.items()}
    return out


def seat_of(key):
    return int(key[len("[Observer: "):key.index("]")])


def merge(tables_by_seat):
    """One C++-ready dict with each seat's entries taken from its own table."""
    out = {}
    for seat, tab in tables_by_seat.items():
        for k, d in tab.items():
            if seat_of(k) == seat:
                out[k] = list(d.items())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfr", default=None)
    ap.add_argument("--alphas", nargs="+", type=float,
                    default=[0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0])
    ap.add_argument("--save-br", nargs="*", type=float, default=[0.1, 0.35],
                    help="alphas at which to pickle the 16 best-response tables")
    ap.add_argument("--cache", default="q1_results.json")
    args = ap.parse_args()

    base = pickle.load(open(find_cfr(args.cfr), "rb"))["table"]
    game = pyspiel.load_game("leduc_poker", {"players": 3})
    root = game.new_initial_state()
    cache = json.load(open(args.cache)) if os.path.exists(args.cache) else {}
    br = pyspiel.TabularBestResponse(game, SEAT, merge({0: base, 1: base, 2: base}))

    t0 = time.time()
    for alpha in args.alphas:
        tabs = {t: perturb(base, t, alpha) for t in TYPES}
        save = alpha in args.save_br
        br_tables = {}
        brfile = f"br_tables_a{alpha}.pkl"
        for t1, t2 in itertools.product(TYPES, repeat=2):
            key = f"{alpha}|{t1}{t2}"
            if key in cache and (not save or os.path.exists(brfile)):
                continue
            opp = {1: tabs[t1], 2: tabs[t2]}
            br.set_policy(merge({0: base, **opp}))
            ceiling = br.value_from_state(root)
            acts = br.get_best_response_actions()

            cfr_pol = pyspiel.TabularPolicy(merge({0: base, **opp}))
            cfr_ret = pyspiel.expected_returns(root, cfr_pol, -1, True)

            br_seat0 = {k: [(a, 1.0)] for k, a in acts.items()}
            br_pol = pyspiel.TabularPolicy({**merge(opp), **br_seat0})
            br_ret = pyspiel.expected_returns(root, br_pol, -1, True)

            cache[key] = {"br": ceiling, "cfr": cfr_ret[SEAT],
                          "cfr_ret": list(cfr_ret), "br_ret": list(br_ret)}
            if save:
                br_tables[f"{t1}{t2}"] = dict(acts)
            json.dump(cache, open(args.cache, "w"))
        if save and br_tables:
            pickle.dump(br_tables, open(brfile, "wb"))
        print(f"alpha {alpha:<5} done  [{time.time()-t0:.0f}s]", flush=True)

    # ------------------------------------------------------------ report
    alphas = sorted({float(k.split("|")[0]) for k in cache})
    for metric, label in [("br", "BEST RESPONSE payout, seat 0"),
                          ("cfr", "CFR payout, seat 0 (no exploitation)")]:
        print(f"\n{label}")
        print(f"{'pair':<6}" + "".join(f"{a:>8}" for a in alphas))
        for t1, t2 in itertools.product(TYPES, repeat=2):
            row = [cache.get(f"{a}|{t1}{t2}", {}).get(metric) for a in alphas]
            print(f"{t1+t2:<6}" + "".join(f"{v:>+8.3f}" if v is not None else f"{'':>8}"
                                          for v in row))


if __name__ == "__main__":
    main()
