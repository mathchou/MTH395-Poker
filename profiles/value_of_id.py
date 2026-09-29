"""
How much is it worth to know who you are playing?

Two best responses, both exact:

  known   Best response to one specific opponent pair. The agent is told who it
          faces. This is the exploitation ceiling for that pair.
  blind   Best response to the DISTRIBUTION over pairs. The agent knows the menu
          of opponent types and their frequencies, but not which pair is seated.

For each pair, known - blind is the value of identification: the most that any
opponent model, however good, could add over a policy that simply plays well
against the population. It is an upper bound on what OSM can buy.

Opponent types in the two seats are drawn independently, uniformly from TYPES.

    python value_of_id.py                      # default 3 types, 9 pairs
    python value_of_id.py --types BASE RAISE:0.35 FOLD:0.35

The blind best response needs a single behaviour strategy per opponent seat that
is equivalent to "draw a type, then play it". Naively averaging the types'
action probabilities at each information set is WRONG: a type that rarely
reaches an information set would get equal say there. The correct mixture
weights each type by its own probability of reaching that information set
(Kuhn's theorem):

    mix(a | I) = sum_k w_k r_k(I) sigma_k(a | I) / sum_k w_k r_k(I)

where r_k(I) is the product of type k's OWN action probabilities on the path to
I. The script checks this is right: the blind BR's value against the mixture
must equal the average of its values against the individual pairs.
"""

import argparse
import itertools
import json
import os
import time

import numpy as np
import pyspiel
from open_spiel.python.algorithms import best_response as br_lib

from best_vs_pair import ProfileFactory, TablePolicy, Joint, exact_returns

SEAT = 0


def own_reach(game, policy, player):
    """r(I) for every information set of `player`: product of its own action probs."""
    reach = {}

    def rec(s, r):
        if s.is_terminal():
            return
        if s.is_chance_node():
            for a, _ in s.chance_outcomes():
                rec(s.child(a), r)
            return
        p = s.current_player()
        if p == player:
            key = s.information_state_string(p)
            reach.setdefault(key, r)       # identical for all histories in I
            dist = policy.action_probabilities(s, p)
            for a in s.legal_actions():
                rec(s.child(a), r * dist.get(a, 0.0))
        else:
            for a in s.legal_actions():
                rec(s.child(a), r)

    rec(game.new_initial_state(), 1.0)
    return reach


def mixture_table(game, policies, weights, player, naive=False):
    """Behaviour strategy equivalent to drawing one of `policies` for `player`."""
    reaches = [own_reach(game, pol, player) for pol in policies]
    table = {}
    for key in reaches[0]:
        num, den = {}, 0.0
        for pol, w, rk in zip(policies, weights, reaches):
            wr = w if naive else w * rk[key]
            dist = pol.table[key]
            for a, p in dist.items():
                num[a] = num.get(a, 0.0) + wr * p
            den += wr
        if den > 0:
            table[key] = {a: v / den for a, v in num.items()}
        else:                               # unreachable by every type
            dist = policies[0].table[key]
            table[key] = {a: 1.0 / len(dist) for a in dist}
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfr", default="cfr_3p.pkl")
    ap.add_argument("--types", nargs="+",
                    default=["RAISE:0.35", "FOLD:0.35", "CALL:0.35"])
    ap.add_argument("--naive", action="store_true",
                    help="also compute the WRONG unweighted mixture, for comparison")
    ap.add_argument("--cache", default="value_of_id.json")
    args = ap.parse_args()

    factory = ProfileFactory(args.cfr)
    game = pyspiel.load_game("leduc_poker", {"players": factory.num_players})
    base = factory.get("BASE")
    types = [t.upper() for t in args.types]
    pols = [factory.get(t) for t in types]
    w = [1.0 / len(types)] * len(types)
    pairs = list(itertools.product(range(len(types)), repeat=2))

    cache = json.load(open(args.cache)) if os.path.exists(args.cache) else {}
    t0 = time.time()

    # ---------------------------------------------------------- blind BR
    print(f"types: {types}   ({len(pairs)} pairs, uniform)\n", flush=True)
    mix = [mixture_table(game, pols, w, p) for p in (1, 2)]
    blind = br_lib.BestResponsePolicy(
        game, player_id=SEAT,
        policy=Joint([base, TablePolicy(mix[0]), TablePolicy(mix[1])]))
    blind_value = blind.value(game.new_initial_state())
    print(f"blind BR value vs the type distribution: {blind_value:+.4f}  "
          f"[{time.time()-t0:.0f}s]", flush=True)

    if args.naive:
        nmix = [mixture_table(game, pols, w, p, naive=True) for p in (1, 2)]
        nb = br_lib.BestResponsePolicy(
            game, player_id=SEAT,
            policy=Joint([base, TablePolicy(nmix[0]), TablePolicy(nmix[1])]))
        print(f"naive-mixture BR thinks it earns:       "
              f"{nb.value(game.new_initial_state()):+.4f}", flush=True)
        naive_actual = np.mean([
            exact_returns(game, [nb, pols[i], pols[j]])[SEAT] for i, j in pairs])
        print(f"naive-mixture BR actually earns:        {naive_actual:+.4f}\n",
              flush=True)

    # ------------------------------------------------------ per pair
    print(f"\n{'pair':<24} {'known':>8} {'blind':>8} {'value of ID':>12} {'cfr':>8}")
    rows = []
    for i, j in pairs:
        name = f"{types[i]} / {types[j]}"
        if name not in cache:
            known = br_lib.BestResponsePolicy(
                game, player_id=SEAT,
                policy=Joint([base, pols[i], pols[j]])).value(game.new_initial_state())
            blind_ev = exact_returns(game, [blind, pols[i], pols[j]])[SEAT]
            cfr_ev = exact_returns(game, [base, pols[i], pols[j]])[SEAT]
            cache[name] = {"known": known, "blind": blind_ev, "cfr": cfr_ev}
            json.dump(cache, open(args.cache, "w"), indent=1)
        r = cache[name]
        rows.append(r)
        print(f"{name:<24} {r['known']:>+8.4f} {r['blind']:>+8.4f} "
              f"{r['known'] - r['blind']:>+12.4f} {r['cfr']:>+8.4f}", flush=True)

    known = np.mean([r["known"] for r in rows])
    blind_avg = np.mean([r["blind"] for r in rows])
    cfr = np.mean([r["cfr"] for r in rows])
    print(f"\n{'average':<24} {known:>+8.4f} {blind_avg:>+8.4f} "
          f"{known - blind_avg:>+12.4f} {cfr:>+8.4f}")
    print(f"\ncorrectness check: blind BR value {blind_value:+.4f} "
          f"vs average over pairs {blind_avg:+.4f}  "
          f"(diff {abs(blind_value - blind_avg):.2e})")
    print(f"\nof the {known - cfr:+.4f} available over CFR, playing well against the "
          f"population\nalready captures {blind_avg - cfr:+.4f} "
          f"({100 * (blind_avg - cfr) / (known - cfr):.0f}%); identifying the pair "
          f"adds the other {known - blind_avg:+.4f}.")
    print(f"[{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
