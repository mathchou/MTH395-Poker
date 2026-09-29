"""
Step 2: build opponent profiles by perturbing the CFR policy, and measure them.

The idea: start from a near-equilibrium policy and tilt it toward one action.

    sigma'(a | I) = (1 - eps) * sigma(a | I) + eps * [a == favoured]

applied only at information sets where the favoured action is legal, and
renormalised. Everywhere else the policy is left alone.

Why this beats hand-written archetypes:

  - The opponents are COMPETENT. A hand-written MANIAC loses 1.5 chips/hand, so
    beating it proves nothing. A perturbed-CFR opponent loses a little, which makes
    exploitation a subtle problem rather than a free lunch.
  - eps is a CONTROLLED VARIABLE. "Distance from equilibrium" becomes a dial you
    turn, not a property you assert about a rule you wrote.
  - The exploitation ceiling is computable per profile: the best-response value
    against a profile is exactly what that deviation costs, and it upper-bounds any
    exploiter you train.

    python make_profiles.py
"""

import pickle
import sys
import time

import numpy as np
import pyspiel
from open_spiel.python.algorithms import best_response as br_lib

FOLD, CALL, RAISE = 0, 1, 2
ACTION_NAME = {FOLD: "FOLD", CALL: "CALL", RAISE: "RAISE"}
NUM_PLAYERS = 3


def probs(policy, state, player_id):
    try:
        return policy.action_probabilities(state, player_id)
    except TypeError:
        return policy.action_probabilities(state)


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
        return probs(self.policies[player_id], state, player_id)


def perturb(table, favoured, eps):
    """Tilt toward `favoured` wherever it is legal. Returns a new table."""
    out = {}
    touched = 0
    for key, dist in table.items():
        if favoured not in dist:
            out[key] = dict(dist)
            continue
        touched += 1
        new = {a: (1 - eps) * p for a, p in dist.items()}
        new[favoured] += eps
        tot = sum(new.values())
        out[key] = {a: p / tot for a, p in new.items()}
    return out, touched


def exact_returns(game, policies):
    n = game.num_players()

    def rec(state, reach):
        if state.is_terminal():
            return np.array(state.returns()) * reach
        if state.is_chance_node():
            return sum(rec(state.child(a), reach * p)
                       for a, p in state.chance_outcomes())
        cur = state.current_player()
        tot = np.zeros(n)
        for a, p in probs(policies[cur], state, cur).items():
            if p > 0:
                tot += rec(state.child(a), reach * p)
        return tot

    return rec(game.new_initial_state(), 1.0)


def measure_style(game, table, n_hands=4000, seed=0):
    """VPIP / PFR / AFq for seat 0 under this policy, opponents playing the same."""
    rng = np.random.default_rng(seed)
    pol = TablePolicy(table)
    num = {"vpip": 0.0, "pfr": 0.0, "afq": 0.0}
    den = {"vpip": 0.0, "pfr": 0.0, "afq": 0.0}
    for _ in range(n_hands):
        s = game.new_initial_state()
        while not s.is_terminal():
            if s.is_chance_node():
                a_, p_ = zip(*s.chance_outcomes())
                s.apply_action(int(rng.choice(a_, p=p_)))
                continue
            p = s.current_player()
            d = probs(pol, s, p)
            ks = list(d)
            a = int(rng.choice(ks, p=[d[k] for k in ks]))
            if p == 0:
                legal = s.legal_actions(p)
                facing = FOLD in legal
                if s.round() == 1:
                    # a free check is not voluntary money
                    num["vpip"] += 1.0 if (a == RAISE or (facing and a == CALL)) else 0.0
                    den["vpip"] += 1.0
                    num["pfr"] += 1.0 if a == RAISE else 0.0
                    den["pfr"] += 1.0
                elif a in (CALL, RAISE):
                    num["afq"] += 1.0 if a == RAISE else 0.0
                    den["afq"] += 1.0
            s.apply_action(a)
    return {k: num[k] / max(den[k], 1) for k in num}


def main():
    with open("cfr_3p.pkl", "rb") as f:
        blob = pickle.load(f)
    base = blob["table"]
    game = pyspiel.load_game("leduc_poker", {"players": blob["players"]})
    print(f"base CFR policy: {len(base):,} infostates, nash_conv {blob['nash_conv']:.4f}")

    base_pol = TablePolicy(base)
    print(f"base style (vpip, pfr, afq): "
          + ", ".join(f"{v:.3f}" for v in measure_style(game, base).values()))
    print(f"base EV seat 0 vs two copies of itself: "
          f"{exact_returns(game, [base_pol] * 3)[0]:+.4f}\n")

    specs = [(RAISE, 0.15), (RAISE, 0.35), (FOLD, 0.25), (CALL, 0.35)]
    print(f"{'profile':<16} {'vpip':>6} {'pfr':>6} {'afq':>6} "
          f"{'EV vs base':>11} {'BR ceiling':>11} {'gap':>8}")
    for fav, eps in specs:
        tab, touched = perturb(base, fav, eps)
        pol = TablePolicy(tab)
        style = measure_style(game, tab)
        # profile sits in seats 1 and 2; how does it fare against the equilibrium?
        ev_as_opponent = exact_returns(game, [base_pol, pol, pol])[1]
        # exploitation ceiling: best response against two copies of the profile
        br = br_lib.BestResponsePolicy(game, player_id=0, policy=Joint([pol, pol, pol]))
        ceiling = br.value(game.new_initial_state())
        name = f"{ACTION_NAME[fav]}+{eps:.2f}"
        print(f"{name:<16} {style['vpip']:>6.3f} {style['pfr']:>6.3f} "
              f"{style['afq']:>6.3f} {ev_as_opponent:>+11.4f} {ceiling:>+11.4f} "
              f"{ceiling:>8.4f}", flush=True)


if __name__ == "__main__":
    main()
