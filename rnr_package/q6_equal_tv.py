"""
Putting the three opponent types on a common axis.

alpha is the probability of overriding the optimal action WHERE THE FAVOURED
ACTION IS LEGAL. That is not comparable across types, for two reasons:

  coverage     raising is illegal at the raise cap, folding is illegal with no bet
               to face, calling is always legal
  headroom     the optimal policy already plays the favoured action sometimes, so
               tilting toward it changes less

For a mixture toward a pure action the total-variation distance at one information
set is exactly

    TV(I) = alpha * (1 - sigma*(fav | I))

so both effects are captured by weighting TV by how often each information set is
actually reached. Define the common axis as the EXPECTED NUMBER OF OVERRIDDEN
DECISIONS PER HAND:

    m(type, alpha) = alpha * sum_I P(I) * (1 - sigma*(fav | I))

where P(I) is the probability that the opponent faces I in a hand under full
equilibrium play. Then invert: alpha = m / rate(type).

    python q6_equal_tv.py --m 0.1 0.25 0.5
"""

import argparse
import itertools
import pickle

import numpy as np
import pyspiel

from q1_alpha_sweep import perturb, merge, TYPES, FAV, seat_of
from cfr_path import find_cfr

SEAT_OPP = 1     # measure the deviation rate for an opponent seat


def reach_weights(game, base, player):
    """P(information set reached in a hand) under full equilibrium play."""
    w = {}

    def rec(s, p):
        if s.is_terminal() or p == 0.0:
            return
        if s.is_chance_node():
            for a, q in s.chance_outcomes():
                rec(s.child(a), p * q)
            return
        cur = s.current_player()
        key = s.information_state_string(cur)
        if cur == player:
            w[key] = w.get(key, 0.0) + p
        d = base[key]
        for a in s.legal_actions():
            rec(s.child(a), p * d.get(a, 0.0))

    rec(game.new_initial_state(), 1.0)
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfr", default=None)
    ap.add_argument("--m", nargs="+", type=float, default=[0.1, 0.25, 0.5])
    args = ap.parse_args()

    base = pickle.load(open(find_cfr(args.cfr), "rb"))["table"]
    game = pyspiel.load_game("leduc_poker", {"players": 3})
    root = game.new_initial_state()

    W = reach_weights(game, base, SEAT_OPP)
    decisions = sum(W.values())
    print(f"expected decisions per hand for an opponent seat: {decisions:.3f}\n")

    rate = {}
    for t, f in FAV.items():
        rate[t] = sum(p * (1 - base[k].get(f, 1.0)) for k, p in W.items() if f in base[k])
    print("deviation rate per unit alpha (overridden decisions per hand):")
    for t in FAV:
        print(f"  {t}: {rate[t]:.4f}   -> alpha = m / {rate[t]:.4f}")

    print(f"\nalpha needed for a target override rate m:")
    print(f"{'m':>6}" + "".join(f"{t:>9}" for t in FAV))
    alphas = {}
    for m in args.m:
        alphas[m] = {t: min(m / rate[t], 1.0) for t in FAV}
        print(f"{m:>6.2f}" + "".join(f"{alphas[m][t]:>9.3f}" for t in FAV))

    br = pyspiel.TabularBestResponse(game, 0, merge({0: base, 1: base, 2: base}))
    base_val = pyspiel.expected_returns(
        root, pyspiel.TabularPolicy(merge({0: base, 1: base, 2: base})), -1, True)[0]

    print(f"\nBR gain over equilibrium at MATCHED override rate "
          f"(seat 0, equilibrium value {base_val:+.4f})")
    print(f"{'m':>6}" + "".join(f"{'O'+t:>9}" for t in FAV)
          + "".join(f"{t+t:>9}" for t in FAV))
    for m in args.m:
        row = []
        for pair in [("O", t) for t in FAV] + [(t, t) for t in FAV]:
            tabs = {t: perturb(base, t, alphas[m][t]) if t != "O" else base
                    for t in TYPES}
            br.set_policy(merge({0: base, 1: tabs[pair[0]], 2: tabs[pair[1]]}))
            row.append(br.value_from_state(root) - base_val)
        print(f"{m:>6.2f}" + "".join(f"{v:>+9.3f}" for v in row), flush=True)


if __name__ == "__main__":
    main()
