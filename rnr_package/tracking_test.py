"""
Known-answer test for identifiers that let old evidence fade.

Seat 0 plays equilibrium and watches. Seat 1 switches style at a given hand (e.g. from an
over-folder to an over-raiser); seat 2 keeps one style. Several identifiers watch the SAME
hands (the expensive per-hand evidence is computed once and shared):

    none          stationary: all evidence counts forever
    forget M      exponential forgetting with memory ~ M hands   (forget = 1 - 1/M)
    switch M      fixed-share, opponents expected to change about every M hands  (switch = 1/M)

For each, per session:
    first     hands until P(true types) >= 0.9 at the start           (cost of forgetting: slower)
    steady    average P(true types) over the hands just before the switch (cost: less certain)
    recover   hands after the switch until P(new true types) >= 0.9   (benefit: faster)

    python tracking_test.py                                    # F:0.35 -> R:0.35 at hand 200
    python tracking_test.py --before F:0.35 --after C:0.35 --seat2 R:0.2 --at 150 --hands 400
"""

import argparse
import pickle
import time

import numpy as np
import pyspiel

from cfr_path import find_cfr
from q1_alpha_sweep import perturb
from q2_grid_identify import GridIdentifier
from q2_identify import play_hand


def style(base, spec):
    t, _, a = spec.partition(":")
    return base if t == "O" or float(a or 0) == 0 else perturb(base, t, float(a))


def first_at(series, start, thresh=0.9):
    for n, v in enumerate(series[start:], 1):
        if v >= thresh:
            return n
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", default="F:0.35"); ap.add_argument("--after", default="R:0.35")
    ap.add_argument("--seat2", default="C:0.3"); ap.add_argument("--at", type=int, default=200)
    ap.add_argument("--hands", type=int, default=400); ap.add_argument("--sessions", type=int, default=10)
    ap.add_argument("--memories", type=int, nargs="+", default=[25, 50, 100, 200])
    ap.add_argument("--tremble", type=float, default=0.01)
    ap.add_argument("--regime", default="showdown"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    game = pyspiel.load_game("leduc_poker", {"players": 3})
    base = pickle.load(open(find_cfr(), "rb"))["table"]
    t1, t1b, t2 = a.before.split(":")[0], a.after.split(":")[0], a.seat2.split(":")[0]
    tables = {"before": [base, style(base, a.before), style(base, a.seat2)],
              "after": [base, style(base, a.after), style(base, a.seat2)]}
    methods = [("none", {})] + [(f"forget {m}", {"forget": 1 - 1 / m}) for m in a.memories] + \
              [(f"switch {m}", {"switch": 1 / m}) for m in a.memories]
    rng = np.random.default_rng(a.seed)
    res = {name: {"first": [], "steady": [], "recover": []} for name, _ in methods}
    t0 = time.time()
    for _ in range(a.sessions):
        ids = [(name, GridIdentifier(game, base, tremble=a.tremble, **kw)) for name, kw in methods]
        series = {name: [] for name, _ in methods}
        for n in range(1, a.hands + 1):
            hist, cards, sd, _ = play_hand(game, tables["before" if n <= a.at else "after"], rng)
            ev = ids[0][1].evidence(hist)                      # computed once, shared by all
            for name, gi in ids:
                gi.absorb(ev, cards, sd)
                P = gi.posterior(a.regime)
                series[name].append(gi.p_types(P, t1 if n <= a.at else t1b, t2))
        for name, _ in methods:
            s = series[name]
            res[name]["first"].append(first_at(s[:a.at], 0))
            res[name]["steady"].append(float(np.mean(s[max(0, a.at - 50):a.at])))
            res[name]["recover"].append(first_at(s, a.at))
    med = lambda v, cap: f"{np.median([x if x is not None else cap for x in v]):.0f}" + \
                         (f" ({sum(x is None for x in v)} never)" if any(x is None for x in v) else "")
    print(f"\nseat 1: {a.before} -> {a.after} at hand {a.at}; seat 2: {a.seat2}; {a.sessions} sessions "
          f"[{time.time() - t0:.0f}s]")
    print(f"{'identifier':<14}{'first (hands)':>16}{'steady P(true)':>16}{'recover (hands)':>18}")
    for name, _ in methods:
        r = res[name]
        print(f"{name:<14}{med(r['first'], a.at):>16}{np.median(r['steady']):>16.3f}"
              f"{med(r['recover'], a.hands - a.at):>18}")


if __name__ == "__main__":
    main()
