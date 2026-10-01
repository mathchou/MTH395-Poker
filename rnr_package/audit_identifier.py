"""
Independent audit of the Bayesian opponent identifier in q2_identify.py.

For random hands, recompute the posterior over the 16 pair hypotheses by brute force:
enumerate every possible opponent deal, weight it with OpenSpiel's OWN chance
probabilities, and include OUR action probabilities too. The identifier drops both
(they cancel in Bayes' rule) and assumes a uniform card prior; if any of that is wrong,
the two posteriors disagree.

Checks:
  1. deal order: history[p] really is seat p's private card
  2. perturb() in q2_identify.py matches perturb() in q1_alpha_sweep.py
  3. brute-force posterior == identifier posterior, all three regimes
  4. with alpha = 0 every hypothesis is identical, so the posterior must stay uniform

    python audit_identifier.py
"""

import itertools
import os
import pickle

import numpy as np
import pyspiel

import q2_identify as q2
import q1_alpha_sweep as q1
from cfr_path import find_cfr


def load_base():
    return pickle.load(open(find_cfr(), "rb"))["table"]


def brute_posterior(game, tables, hist, regime, true_cards, showdown):
    """P(pair | observations) by enumerating deals with OpenSpiel's chance probabilities."""
    c0 = hist[0]
    deck = [a for a, _ in game.new_initial_state().chance_outcomes()]
    like = np.zeros(16)
    for c1, c2 in itertools.permutations([c for c in deck if c != c0], 2):
        if regime == "critic" and (c1, c2) != (true_cards[1], true_cards[2]):
            continue
        if regime == "showdown" and ((1 in showdown and c1 != true_cards[1]) or
                                     (2 in showdown and c2 != true_cards[2])):
            continue
        h = list(hist); h[1], h[2] = c1, c2
        for k, (t1, t2) in enumerate(itertools.product(q2.TYPES, repeat=2)):
            pol = [tables["O"], tables[t1], tables[t2]]
            st, pr = game.new_initial_state(), 1.0
            for a in h:
                if st.is_chance_node():
                    pr *= dict(st.chance_outcomes()).get(a, 0.0)   # 0 if the board clashes
                else:
                    p = st.current_player()
                    pr *= pol[p][st.information_state_string(p)].get(a, 0.0)
                if pr == 0.0:
                    break
                st.apply_action(a)
            like[k] += pr
    return like / like.sum()


def main():
    game = pyspiel.load_game("leduc_poker", {"players": 3})
    base = load_base()
    rng = np.random.default_rng(0)

    # 1. deal order
    ok = True
    for _ in range(50):
        hist, cards, _, _ = q2.play_hand(game, [base] * 3, rng)
        st = game.new_initial_state()
        for i in range(3):
            assert st.is_chance_node()
            st.apply_action(hist[i])
        for p in range(3):
            alt = game.new_initial_state()
            h = list(hist[:3])
            swap = next(c for c in range(len(h) + 5) if c not in h)   # a different card
            h[p] = swap
            for a in h:
                alt.apply_action(a)
            ok &= alt.information_state_string(p) != st.information_state_string(p)
            ok &= all(alt.information_state_string(q) == st.information_state_string(q) for q in range(3) if q != p)
    print(f"1. deal order: history[p] is seat p's card ............ {'PASS' if ok else 'FAIL'}")

    # 2. perturb copies agree
    worst = 0.0
    for t in "RCF":
        for a in (0.1, 0.35, 1.0):
            A, B = q2.perturb(base, t, a), q1.perturb(base, t, a)
            worst = max(worst, max(abs(A[k].get(x, 0) - B[k].get(x, 0)) for k in A for x in set(A[k]) | set(B[k])))
    print(f"2. perturb() copies agree (max difference {worst:.1e}) ..... {'PASS' if worst < 1e-12 else 'FAIL'}")

    # 3. brute force vs identifier
    alpha = 0.35
    tables = {t: q2.perturb(base, t, alpha) for t in q2.TYPES}
    worst = {r: 0.0 for r in q2.REGIMES}
    for i in range(12):
        tp = q2.PAIRS[rng.integers(16)]
        hist, cards, sd, _ = q2.play_hand(game, [base, tables[tp[0]], tables[tp[1]]], rng)
        idf = q2.Identifier(game, tables)
        idf.update(hist, cards, sd)
        for r in q2.REGIMES:
            worst[r] = max(worst[r], np.abs(idf.posterior(r) - brute_posterior(game, tables, hist, r, cards, sd)).max())
    for r in q2.REGIMES:
        print(f"3. brute force == identifier, {r:<8} (max diff {worst[r]:.1e}) {'PASS' if worst[r] < 1e-10 else 'FAIL'}")

    # 4. alpha = 0: nothing to learn
    tables0 = {t: q2.perturb(base, t, 0.0) for t in q2.TYPES}
    idf = q2.Identifier(game, tables0)
    for _ in range(30):
        hist, cards, sd, _ = q2.play_hand(game, [base] * 3, rng)
        idf.update(hist, cards, sd)
    dev = max(np.abs(idf.posterior(r) - 1 / 16).max() for r in q2.REGIMES)
    print(f"4. alpha = 0 keeps the posterior uniform (max dev {dev:.1e}) {'PASS' if dev < 1e-12 else 'FAIL'}")


if __name__ == "__main__":
    main()
