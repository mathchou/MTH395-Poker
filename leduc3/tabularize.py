"""
Convert a neural policy into a tabular open_spiel policy.

Calling a torch forward pass at every node of a 1.8M-node tree makes nash_conv
impractically slow. Enumerating the 25,800 information states once and caching the
action distribution turns exploitability measurement back into a tree walk over a
dict. This is required infrastructure for Experiment E, not an optimisation.
"""
import time
import numpy as np
import pyspiel


class TabularPolicy(pyspiel.Policy):
    def __init__(self, table):
        super().__init__()
        self.table = table

    def action_probabilities(self, state, player_id=None):
        if player_id is None:
            player_id = state.current_player()
        return self.table[state.information_state_string(player_id)]


def tabularize(game, policy, verbose=True):
    table, seen = {}, set()
    t0 = time.time()

    def rec(state):
        if state.is_terminal():
            return
        if state.is_chance_node():
            for a, _ in state.chance_outcomes():
                rec(state.child(a))
            return
        p = state.current_player()
        key = state.information_state_string(p)
        if key not in seen:
            seen.add(key)
            table[key] = policy.action_probabilities(state, p)
        for a in state.legal_actions():
            rec(state.child(a))

    rec(game.new_initial_state())
    if verbose:
        print(f"     tabularized {len(table):,} info states in {time.time()-t0:.1f}s")
    return TabularPolicy(table)
