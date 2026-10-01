"""
Audit of the CFR+ solver in rnr_general.py against fixed opponents.

At p = 1 both opponents always follow their biased models, so the restricted-response
game is just "CFR+ for our seat against two fixed opponents", and its answer is known
independently: OpenSpiel's best-response value against those same opponents.

Checks:
  1. our solved strategy's value approaches OpenSpiel's best-response value
  2. the solver's reported convergence gap equals (OpenSpiel BR value - our value)
  3. the engine's own best response (tree_engine.respond) equals OpenSpiel's

    python audit_solver.py            # about 2-4 minutes
"""

import json
import os
import pickle
import types

import numpy as np
import pyspiel

import rnr_general as rg
from q1_alpha_sweep import perturb, merge
from tree_engine import Tree
from cfr_path import find_cfr

PAIR, ALPHA, SEAT, ITERS = "FF", 0.35, 0, 150


def main():
    cfr = find_cfr()                 # the same file rnr_general.load() reads
    blob = pickle.load(open(cfr, "rb"))
    print(f"rnr_general reads {cfr}: {blob.get('iters')} iterations")
    base = blob["table"]

    args = types.SimpleNamespace(pair=PAIR, alpha=ALPHA, seat=SEAT, p=1.0, restrict="independent",
                                 free="selfish", iters=ITERS, suffix="audit_p1", seed=None)
    rg.solve(args)
    tag = f"{PAIR}_a{ALPHA}_s{SEAT}_independent_selfish_p1.0_audit_p1"
    res = json.load(open(os.path.join(rg.OUT, f"rnr_{tag}.json")))
    strat = np.load(os.path.join(rg.OUT, f"rnr_{tag}_strategy.npy"))

    T = Tree()
    opps = [o for o in range(3) if o != SEAT]
    tabs = {o: perturb(base, t, ALPHA) for o, t in zip(opps, PAIR)}
    prof = [None] * 3
    for o in opps:
        prof[o] = T.from_table(tabs[o], o)
    prof[SEAT] = strat
    ours = T.ev(prof)[SEAT]

    game = pyspiel.load_game("leduc_poker", {"players": 3})
    pol = merge({SEAT: base, **tabs})
    os_br = pyspiel.TabularBestResponse(game, SEAT, pol).value_from_state(game.new_initial_state())
    prof[SEAT] = None
    _, eng_br = T.respond(prof, SEAT, T.util[:, SEAT], T.util[:, SEAT])

    print(f"\nafter {ITERS} iterations at p = 1 ({PAIR}, alpha {ALPHA}, seat {SEAT}):")
    print(f"  our strategy's value          {ours:+.6f}")
    print(f"  OpenSpiel best-response value {os_br:+.6f}")
    print(f"  engine best-response value    {eng_br:+.6f}")
    gap_true = os_br - ours
    print(f"1. solver within {gap_true:.4f} of the best response ........ "
          f"{'PASS' if 0 <= gap_true < 0.01 else 'CHECK'}")
    print(f"2. reported gap {res['gaps']['learner']:.6f} vs true gap {gap_true:.6f} .... "
          f"{'PASS' if abs(res['gaps']['learner'] - gap_true) < 1e-9 else 'FAIL'}")
    print(f"3. engine BR == OpenSpiel BR (diff {abs(eng_br - os_br):.1e}) ......... "
          f"{'PASS' if abs(eng_br - os_br) < 1e-9 else 'FAIL'}")


if __name__ == "__main__":
    main()
