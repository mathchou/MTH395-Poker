"""
Step 1: solve 3-player Leduc with CFR+ and freeze the average policy to disk.

Everything downstream (perturbed opponent profiles, exploiter training, the
Experiment E lambda=0 anchor) reads this file, so it is computed once.

    python solve_cfr.py 40 cfr_3p.pkl

Note the solver must stay alive while the policy is read: average_policy() returns
a view into its memory, and dropping the solver makes every lookup raise
"No policy found, and no default policy."
"""

import pickle
import sys
import time

import pyspiel


def solve_and_save(n_players=3, iters=40, path="cfr_3p.pkl", log_every=10):
    game = pyspiel.load_game("leduc_poker", {"players": n_players})
    solver = pyspiel.CFRPlusSolver(game)

    print(f"{n_players}-player Leduc, CFR+, {iters} iterations")
    print(f"{'iter':>6} {'nash_conv':>12} {'secs':>8}")
    t0 = time.time()
    for i in range(1, iters + 1):
        solver.evaluate_and_update_policy()
        if i % log_every == 0 or i == iters:
            nc = pyspiel.nash_conv(game, solver.average_policy())
            print(f"{i:>6} {nc:>12.5f} {time.time()-t0:>8.1f}", flush=True)

    avg = solver.average_policy()          # solver still in scope
    table = {}

    def rec(s):
        if s.is_terminal():
            return
        if s.is_chance_node():
            for a, _ in s.chance_outcomes():
                rec(s.child(a))
            return
        key = s.information_state_string(s.current_player())
        if key not in table:
            table[key] = dict(avg.action_probabilities(s))
        for a in s.legal_actions():
            rec(s.child(a))

    rec(game.new_initial_state())
    final_nc = pyspiel.nash_conv(game, avg)
    with open(path, "wb") as f:
        pickle.dump({"table": table, "nash_conv": final_nc,
                     "iters": iters, "players": n_players}, f)
    print(f"saved {len(table):,} infostates -> {path}  (nash_conv {final_nc:.5f})")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    out = sys.argv[2] if len(sys.argv) > 2 else "cfr_3p.pkl"
    solve_and_save(iters=n, path=out, log_every=50)
