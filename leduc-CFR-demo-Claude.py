"""
CFR on Leduc Hold'em using pyspiel's built-in C++ solvers.

    python cfr_demo.py 2          # 2-player, converges in seconds
    python cfr_demo.py 3 20       # 3-player, ~4.3 s per iteration

Two gotchas, both of which cost real time to find:

  1. average_policy() returns a view into the solver's memory. If the solver is
     garbage-collected, every lookup raises "No policy found, and no default
     policy." Keep a reference to the solver for as long as you use the policy.
  2. pyspiel.exploitability() asserts a 2-player constant-sum game. For three or
     more players use pyspiel.nash_conv(), which is the sum over players of the
     gain from a unilateral best response — exactly the no-collusion metric.
"""

import pickle
import sys
import time

import pyspiel


def solve(n_players, iters, log_every, solver_name="CFRPlus"):
    game = pyspiel.load_game("leduc_poker", {"players": n_players})
    solver = {"CFR": pyspiel.CFRSolver,
              "CFRPlus": pyspiel.CFRPlusSolver}[solver_name](game)

    print(f"\n{n_players}-player Leduc, {solver_name}")
    print(f"{'iter':>6} {'nash_conv':>12} {'secs':>8}")
    t0 = time.time()
    for i in range(1, iters + 1):
        solver.evaluate_and_update_policy()
        if i % log_every == 0 or i == iters:
            nc = pyspiel.nash_conv(game, solver.average_policy())
            print(f"{i:>6} {nc:>12.5f} {time.time()-t0:>8.1f}", flush=True)

    # Return the solver too. Dropping it invalidates the policy. See gotcha 1.
    return game, solver, solver.average_policy()


def save_tabular(game, avg, path):
    """Freeze the policy into a plain dict of infostate -> action probabilities."""
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
            table[key] = avg.action_probabilities(s)
        for a in s.legal_actions():
            rec(s.child(a))

    rec(game.new_initial_state())
    with open(path, "wb") as f:
        pickle.dump(table, f)
    print(f"saved {len(table):,} infostates -> {path}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    if n == 2:
        game, solver, avg = solve(2, 200, 25)
    else:
        iters = int(sys.argv[2]) if len(sys.argv) > 2 else 20
        game, solver, avg = solve(3, iters, max(iters // 4, 1))
    save_tabular(game, avg, f"cfr_{n}p.pkl")