"""
Minimal A2C on open_spiel, 3-player Leduc Hold'em.

Self-contained and readable — about 90 lines of actual logic. The agent sits in
seat 0 and learns against two fixed scripted opponents.

Three things that are specific to imperfect-information games and easy to get wrong:

  1. ACTION MASKING. Legal actions change with the state. Illegal logits must be set
     to -inf BEFORE the softmax, not zeroed after, or gradients leak into illegal
     actions.
  2. INFORMATION STATE, not state. The agent must be fed
     state.information_state_tensor(player), which hides other players' cards. Using
     observation_tensor or the raw state leaks private information.
  3. NO INTERMEDIATE REWARD. Poker pays out only at terminal states, and a Leduc hand
     is at most about six decisions, so plain Monte Carlo returns are fine. Every
     decision in a hand gets that hand's final return. Bootstrapping buys nothing at
     this horizon.
"""

import numpy as np
import pyspiel
import torch
import torch.nn as nn

NUM_PLAYERS = 3
AGENT_SEAT = 0
FOLD, CALL, RAISE = 0, 1, 2


# --------------------------------------------------------------------- networks

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.pi = nn.Linear(hidden, n_actions)
        self.v = nn.Linear(hidden, 1)

    def forward(self, obs, legal_mask):
        h = self.body(obs)
        logits = self.pi(h).masked_fill(~legal_mask, float("-inf"))   # point 1
        return logits, self.v(h).squeeze(-1)


# -------------------------------------------------------------------- opponents

def scripted_opponent(state, player, rng):
    """Tight-ish rule: raise strong, call medium, fold weak. Slightly randomised."""
    legal = state.legal_actions(player)
    rank = state.private_card(player) // 2
    pub = state.public_card()
    if pub is not None and pub >= 0 and rank == pub // 2:
        strength = 1.0                       # paired the board
    else:
        strength = rank / 3.0                # 4 ranks in the 3-player deck
    if strength > 0.7 and RAISE in legal:
        probs = {RAISE: 0.8, CALL: 0.2}
    elif strength > 0.3:
        probs = {CALL: 0.85, RAISE: 0.15}
    else:
        probs = {FOLD: 0.6, CALL: 0.4} if FOLD in legal else {CALL: 1.0}
    probs = {a: p for a, p in probs.items() if a in legal}
    tot = sum(probs.values())
    acts = list(probs)
    return int(rng.choice(acts, p=[probs[a] / tot for a in acts]))


# ------------------------------------------------------------------- rollouts

def play_hand(net, game, rng):
    """One hand. Returns (observations, masks, actions, final_return)."""
    state = game.new_initial_state()
    obs, masks, acts = [], [], []

    while not state.is_terminal():
        if state.is_chance_node():
            outcomes, probs = zip(*state.chance_outcomes())
            state.apply_action(int(rng.choice(outcomes, p=probs)))
            continue

        p = state.current_player()
        if p != AGENT_SEAT:
            state.apply_action(scripted_opponent(state, p, rng))
            continue

        o = torch.tensor(state.information_state_tensor(p),      # point 2
                         dtype=torch.float32)
        m = torch.zeros(game.num_distinct_actions(), dtype=torch.bool)
        for a in state.legal_actions(p):
            m[a] = True

        with torch.no_grad():
            logits, _ = net(o.unsqueeze(0), m.unsqueeze(0))
            a = int(torch.distributions.Categorical(logits=logits[0]).sample())

        obs.append(o); masks.append(m); acts.append(a)
        state.apply_action(a)

    return obs, masks, acts, state.returns()[AGENT_SEAT]     # point 3


# ---------------------------------------------------------------------- update

def a2c_update(net, opt, batch, ent_coef=0.05):
    obs = torch.stack([b[0] for b in batch])
    masks = torch.stack([b[1] for b in batch])
    acts = torch.tensor([b[2] for b in batch])
    rets = torch.tensor([b[3] for b in batch], dtype=torch.float32)

    logits, values = net(obs, masks)
    dist = torch.distributions.Categorical(logits=logits)

    advantage = (rets - values).detach()
    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

    policy_loss = -(dist.log_prob(acts) * advantage).mean()
    value_loss = nn.functional.mse_loss(values, rets)
    entropy = dist.entropy().mean()

    loss = policy_loss + 0.5 * value_loss - ent_coef * entropy

    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step()
    return float(entropy.detach())


# ------------------------------------------------------------------------ main

def main(updates=300, hands_per_update=128, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    game = pyspiel.load_game("leduc_poker", {"players": NUM_PLAYERS})
    obs_dim = game.information_state_tensor_size()
    net = ActorCritic(obs_dim, game.num_distinct_actions())
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)

    print(f"obs_dim={obs_dim}  actions={game.num_distinct_actions()}", flush=True)

    for u in range(1, updates + 1):
        batch, returns = [], []
        for _ in range(hands_per_update):
            obs, masks, acts, R = play_hand(net, game, rng)
            returns.append(R)
            batch.extend((o, m, a, R) for o, m, a in zip(obs, masks, acts))
        if not batch:
            continue
        ent = a2c_update(net, opt, batch)
        if u % 50 == 0:
            print(f"update {u:4d}  mean return {np.mean(returns):+.3f}  "
                  f"entropy {ent:.3f}", flush=True)


if __name__ == "__main__":
    main()