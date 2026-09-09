"""Walk one hand of Leduc, printing what each player can and cannot see."""
import numpy as np, pyspiel

RANK = "JQKA23"          # rank i = card // 2
def card(c): return "?" if c is None or c < 0 else RANK[c // 2]
ACT = {0: "fold", 1: "call/check", 2: "raise"}

def walk(n_players, seed):
    g = pyspiel.load_game("leduc_poker", {"players": n_players})
    rng = np.random.default_rng(seed)
    s = g.new_initial_state()
    print(f"\n{'='*58}\n{n_players}-PLAYER LEDUC   deck={len(s.chance_outcomes())} cards, "
          f"{len(s.chance_outcomes())//2} ranks\n{'='*58}")
    while not s.is_terminal():
        if s.is_chance_node():
            acts, probs = zip(*s.chance_outcomes())
            c = int(rng.choice(acts, p=probs))
            who = "board" if all(s.private_card(p) >= 0 for p in range(n_players)) else "deal"
            s.apply_action(c)
            print(f"  [chance] {who}: {card(c)}")
            continue
        p = s.current_player()
        legal = s.legal_actions(p)
        print(f"  P{p} holds {card(s.private_card(p))} | board {card(s.public_card())} "
              f"| pot {s.pot():2d} | round {s.round()} | legal {[ACT[a] for a in legal]}")
        a = int(rng.choice(legal))
        s.apply_action(a)
        print(f"     -> P{p} {ACT[a]}")
    print(f"  SHOWDOWN  hands: {[card(s.private_card(p)) for p in range(n_players)]} "
          f"board {card(s.public_card())}")
    print(f"  RETURNS   {s.returns()}  (sums to {sum(s.returns()):.0f})")

walk(2, 3)
walk(3, 3)

# what a player actually sees
g = pyspiel.load_game("leduc_poker", {"players": 3})
s = g.new_initial_state()
for c in [0, 2, 4]: s.apply_action(c)
print(f"\n{'='*58}\nINFORMATION STATE — P0 holds {card(0)}, P1 {card(2)}, P2 {card(4)}\n{'='*58}")
for p in range(3):
    print(f"  P{p} sees: {s.information_state_string(p)}")
print(f"\n  tensor length: {g.information_state_tensor_size()}  "
      f"(note: no other player's card in it)")