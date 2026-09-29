"""
Best policy against any pair of biased opponents.

Each opponent is the CFR policy tilted toward one action:

    sigma'(a | I) = (1 - eps) * sigma_CFR(a | I) + eps * [a == favoured]

applied where the favoured action is legal, then renormalised. An opponent is
written ACTION:EPS, e.g. RAISE:0.35, FOLD:0.2, CALL:0.1. Use BASE for the
unperturbed CFR policy.

Single pair:
    python best_vs_pair.py --opp1 RAISE:0.35 --opp2 FOLD:0.25 --seat 0

Grid over bias degree (every combination of the listed eps for two actions):
    python best_vs_pair.py --sweep RAISE FOLD --eps 0.1 0.3

Save the best-response policy to disk for later play or evaluation:
    python best_vs_pair.py --opp1 RAISE:0.35 --opp2 RAISE:0.35 --save br.pkl

What gets reported:
    ceiling   best-response value: the most any policy in this seat can earn
    cfr_ev    what the unperturbed CFR policy earns in the same seat
    gain      ceiling - cfr_ev: the value of exploiting this pair rather than
              playing the CFR policy
    excess    gain - gain(BASE, BASE): the part of the gain caused by the
              opponents' bias. THIS is the number to compare across pairs.

Why excess and not gain: the base CFR policy is not converged, so even two
unbiased CFR opponents can be exploited. gain(BASE, BASE) is exactly the base
policy's own exploitability in this seat, and it shifts every cell of a sweep by
the same amount. Subtracting cfr_ev does NOT remove it. Only subtracting the
BASE/BASE cell does. With a converged base (nash_conv < 0.01) the two coincide.
"""

import argparse
import pickle
import time

import numpy as np
import pyspiel
from open_spiel.python.algorithms import best_response as br_lib

ACTIONS = {"FOLD": 0, "CALL": 1, "RAISE": 2}


# ------------------------------------------------------------------ policies

class TablePolicy(pyspiel.Policy):
    def __init__(self, table):
        super().__init__()
        self.table = table

    def action_probabilities(self, state, player_id=None):
        if player_id is None:
            player_id = state.current_player()
        return self.table[state.information_state_string(player_id)]


class Joint(pyspiel.Policy):
    def __init__(self, policies):
        super().__init__()
        self.policies = policies

    def action_probabilities(self, state, player_id=None):
        if player_id is None:
            player_id = state.current_player()
        return self.policies[player_id].action_probabilities(state, player_id)


# ---------------------------------------------------------------- profiles

class ProfileFactory:
    """Builds and caches perturbed tables from one base CFR table."""

    def __init__(self, path="cfr_3p.pkl"):
        with open(path, "rb") as f:
            blob = pickle.load(f)
        self.base = blob["table"]
        self.base_nash_conv = blob.get("nash_conv")
        self.num_players = blob.get("players", 3)
        self._cache = {}

    def get(self, spec):
        spec = spec.upper()
        if spec in self._cache:
            return self._cache[spec]
        if spec == "BASE":
            pol = TablePolicy(self.base)
        else:
            name, eps = spec.split(":")
            fav, eps = ACTIONS[name], float(eps)
            if not 0.0 <= eps <= 1.0:
                raise ValueError(f"eps must be in [0, 1], got {eps}")
            table = {}
            for key, dist in self.base.items():
                if fav not in dist:
                    table[key] = dist
                    continue
                new = {a: (1 - eps) * p for a, p in dist.items()}
                new[fav] += eps
                tot = sum(new.values())
                table[key] = {a: p / tot for a, p in new.items()}
            pol = TablePolicy(table)
        self._cache[spec] = pol
        return pol


# -------------------------------------------------------------- evaluation

def exact_returns(game, policies):
    """Exact expected returns by full tree walk. Deterministic, zero variance."""
    n = game.num_players()

    def rec(state, reach):
        if state.is_terminal():
            return np.array(state.returns()) * reach
        if state.is_chance_node():
            return sum(rec(state.child(a), reach * p)
                       for a, p in state.chance_outcomes())
        cur = state.current_player()
        tot = np.zeros(n)
        for a, p in policies[cur].action_probabilities(state, cur).items():
            if p > 0:
                tot += rec(state.child(a), reach * p)
        return tot

    return rec(game.new_initial_state(), 1.0)


def seat_policies(learner, opp1, opp2, seat, num_players=3):
    """Put the learner in `seat`, opponents in the remaining seats in order."""
    others = iter([opp1, opp2])
    return [learner if p == seat else next(others) for p in range(num_players)]


def best_vs_pair(game, factory, spec1, spec2, seat=0, verify=False):
    opp1, opp2 = factory.get(spec1), factory.get(spec2)
    base = factory.get("BASE")

    t0 = time.time()
    # The learner's own entry in the joint policy is ignored by BestResponsePolicy.
    joint = Joint(seat_policies(base, opp1, opp2, seat))
    br = br_lib.BestResponsePolicy(game, player_id=seat, policy=joint)
    ceiling = br.value(game.new_initial_state())

    cfr_ev = exact_returns(game, seat_policies(base, opp1, opp2, seat))[seat]

    out = {"opp1": spec1, "opp2": spec2, "seat": seat,
           "ceiling": ceiling, "cfr_ev": cfr_ev, "gain": ceiling - cfr_ev,
           "policy": br, "secs": time.time() - t0}

    if verify:
        played = exact_returns(game, seat_policies(br, opp1, opp2, seat))[seat]
        out["verify"] = played
    return out


def freeze_br(game, br, seat):
    """Tabularize a BestResponsePolicy. Only the learner's information sets exist."""
    table = {}

    def rec(s):
        if s.is_terminal():
            return
        if s.is_chance_node():
            for a, _ in s.chance_outcomes():
                rec(s.child(a))
            return
        p = s.current_player()
        if p == seat:
            key = s.information_state_string(p)
            if key not in table:
                table[key] = br.action_probabilities(s, p)
        for a in s.legal_actions():
            rec(s.child(a))

    rec(game.new_initial_state())
    return table


# ---------------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfr", default="cfr_3p.pkl")
    ap.add_argument("--opp1", default="RAISE:0.35")
    ap.add_argument("--opp2", default="RAISE:0.35")
    ap.add_argument("--seat", type=int, default=0)
    ap.add_argument("--verify", action="store_true",
                    help="play the best response and confirm it earns the ceiling")
    ap.add_argument("--save", default=None, help="pickle the BR policy table here")
    ap.add_argument("--sweep", nargs=2, metavar=("ACTION1", "ACTION2"),
                    help="grid over eps: opp1 favours ACTION1, opp2 favours ACTION2")
    ap.add_argument("--eps", nargs="+", type=float, default=[0.1, 0.3])
    args = ap.parse_args()

    factory = ProfileFactory(args.cfr)
    game = pyspiel.load_game("leduc_poker", {"players": factory.num_players})
    print(f"base CFR nash_conv {factory.base_nash_conv:.4f}  (seat {args.seat})\n")

    if args.sweep:
        a1, a2 = (a.upper() for a in args.sweep)
        eps_list = [0.0] + [e for e in args.eps if e > 0]
        print(f"gain over CFR (chips/hand)   rows: opp1 {a1}   cols: opp2 {a2}")
        print(f"{'eps':>8}" + "".join(f"{e:>9.2f}" for e in eps_list))
        grid = []
        for e1 in eps_list:
            row = []
            for e2 in eps_list:
                s1 = "BASE" if e1 == 0 else f"{a1}:{e1}"
                s2 = "BASE" if e2 == 0 else f"{a2}:{e2}"
                row.append(best_vs_pair(game, factory, s1, s2, args.seat)["gain"])
            grid.append(row)
            print(f"{e1:>8.2f}" + "".join(f"{g:>+9.4f}" for g in row), flush=True)
        corner = grid[0][0]
        print(f"\nexcess (gain minus the BASE/BASE corner, {corner:+.4f})")
        print(f"{'eps':>8}" + "".join(f"{e:>9.2f}" for e in eps_list))
        for e1, row in zip(eps_list, grid):
            print(f"{e1:>8.2f}" + "".join(f"{g - corner:>+9.4f}" for g in row))
        return

    r = best_vs_pair(game, factory, args.opp1, args.opp2, args.seat, args.verify)
    base_gain = best_vs_pair(game, factory, "BASE", "BASE", args.seat)["gain"]
    print(f"opponents : {r['opp1']}  +  {r['opp2']}")
    print(f"ceiling   : {r['ceiling']:+.4f}   best achievable in seat {r['seat']}")
    print(f"cfr_ev    : {r['cfr_ev']:+.4f}   equilibrium policy in the same seat")
    print(f"gain      : {r['gain']:+.4f}   value of exploiting this pair")
    print(f"excess    : {r['gain'] - base_gain:+.4f}   gain attributable to the bias "
          f"(base alone gives {base_gain:+.4f})")
    if "verify" in r:
        print(f"verify    : {r['verify']:+.4f}   BR actually played (should equal ceiling)")
    print(f"({r['secs']:.1f}s)")

    if args.save:
        table = freeze_br(game, r["policy"], args.seat)
        with open(args.save, "wb") as f:
            pickle.dump({"table": table, "seat": args.seat,
                         "opp1": args.opp1, "opp2": args.opp2,
                         "ceiling": r["ceiling"]}, f)
        print(f"saved {len(table):,} information sets -> {args.save}")


if __name__ == "__main__":
    main()
