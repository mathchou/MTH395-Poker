"""
Vectorized trainer. Same algorithm as train.py, but rolls out many hands in
lockstep so every agent decision across the batch is served by ONE forward pass
instead of one per decision.

train.py does roughly 3 batch-1 forward passes per hand. At 128 hands per update
that is ~384 sequential LSTM calls. Here the batch advances together: at each step
we gather every hand currently waiting on the agent, encode them as a block, and
run a single forward. Chance nodes and scripted-opponent actions stay in Python
where they are cheap.

Also adds entropy annealing. A fixed coefficient cannot be right for the whole run:
0.01 lets the policy collapse early, 0.08 prevents sharp play late.
"""

import argparse
import time

import numpy as np
import torch
import torch.nn as nn

import encode
from styles import make_game, training_styles, ArchetypePolicy, PlayerType, NUM_PLAYERS
from networks import AMP3Network, FULL_STYLE_DIM
from train import PerfectCritic, encode_perfect, measured_style_vector
from evaluate import expected_returns_exact

ZERO_STYLE = np.zeros(FULL_STYLE_DIM, dtype=np.float32)
USE_MEASURED = False


class Hand:
    __slots__ = ("state", "seat", "opps", "seq", "mine", "done")

    def __init__(self, state, seat, opps):
        self.state = state
        self.seat = seat
        self.opps = opps
        self.seq = []
        self.mine = []
        self.done = False


def advance_to_agent(h, rng):
    """Step chance nodes and opponent actions until it is the agent's turn."""
    s = h.state
    while not s.is_terminal():
        if s.is_chance_node():
            a_, p_ = zip(*s.chance_outcomes())
            s.apply_action(int(rng.choice(a_, p=p_)))
            continue
        p = s.current_player()
        if p == h.seat:
            return True
        opp = h.opps[0] if p == (h.seat + 1) % NUM_PLAYERS else h.opps[1]
        pr = opp.action_probabilities(s, p)
        ks = list(pr)
        a = int(rng.choice(ks, p=[pr[k] for k in ks]))
        h.seq.append((p, a))
        s.apply_action(a)
    h.done = True
    return False


def style_for(h):
    return measured_style_vector(h.opps) if USE_MEASURED else ZERO_STYLE


def collect_vec(actor, game, pool, rng, n_hands):
    hands = []
    for i in range(n_hands):
        seat = i % NUM_PLAYERS
        opps = [pool[rng.integers(len(pool))] for _ in range(2)]
        hands.append(Hand(game.new_initial_state(), seat, opps))

    live = [h for h in hands if advance_to_agent(h, rng)]
    while live:
        per = torch.tensor(np.stack([encode.encode_personal(h.state, h.seat) for h in live]))
        pub = torch.tensor(np.stack([encode.encode_public(h.state) for h in live]))
        pos = torch.tensor(np.stack([encode.encode_position(h.seat) for h in live]))
        ah = torch.tensor(np.stack([encode.encode_action_history(h.seq, h.seat) for h in live]))
        st = torch.tensor(np.stack([style_for(h) for h in live]))
        mask = torch.zeros(len(live), encode.NUM_ACTIONS, dtype=torch.bool)
        for i, h in enumerate(live):
            for a in h.state.legal_actions(h.seat):
                mask[i, a] = True

        with torch.no_grad():
            logits, _ = actor(per, pub, pos, ah, st, legal_mask=mask)
            acts = torch.distributions.Categorical(logits=logits).sample()

        for i, h in enumerate(live):
            a = int(acts[i])
            h.mine.append((per[i], pub[i], pos[i], ah[i], st[i], mask[i], a,
                           encode_perfect(h.state, h.seat),
                           encode.encode_position(h.seat)))
            h.seq.append((h.seat, a))
            h.state.apply_action(a)
        live = [h for h in live if advance_to_agent(h, rng)]

    obs, acts_l, rets, perf = [], [], [], []
    for h in hands:
        R = h.state.returns()[h.seat]
        for (p_, u_, po_, ah_, st_, m_, a_, pf_, pos_) in h.mine:
            obs.append((p_, u_, po_, ah_, st_, m_))
            acts_l.append(a_); rets.append(R)
            perf.append(np.concatenate([pf_, pos_]))
    return obs, acts_l, np.array(rets, np.float32), np.array(perf, np.float32)


def update(actor, critic, opt, obs, acts, rets, perf, ent_coef):
    per = torch.stack([o[0] for o in obs]); pub = torch.stack([o[1] for o in obs])
    pos = torch.stack([o[2] for o in obs]); ah = torch.stack([o[3] for o in obs])
    st = torch.stack([o[4] for o in obs]); mask = torch.stack([o[5] for o in obs])
    a = torch.tensor(acts); R = torch.tensor(rets); P = torch.tensor(perf)

    logits, _ = actor(per, pub, pos, ah, st, legal_mask=mask)
    dist = torch.distributions.Categorical(logits=logits)
    V = critic(P)
    adv = (R - V).detach()
    adv = (adv - adv.mean()) / (adv.std() + 1e-6)

    loss = (-(dist.log_prob(a) * adv).mean()
            + 0.5 * nn.functional.mse_loss(V, R)
            - ent_coef * dist.entropy().mean())
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(actor.parameters()) + list(critic.parameters()), 1.0)
    opt.step()
    return float(dist.entropy().mean().detach())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=1000)
    ap.add_argument("--hands", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ent-start", type=float, default=0.08)
    ap.add_argument("--ent-end", type=float, default=0.01)
    ap.add_argument("--ent-total", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default="ck_fast.pt")
    ap.add_argument("--style", choices=["none", "measured"], default="none")
    ap.add_argument("--init-from", default=None)
    args = ap.parse_args()

    global USE_MEASURED
    USE_MEASURED = (args.style == 'measured')
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    game = make_game(); rng = np.random.default_rng(args.seed)
    pool = training_styles()
    actor, critic = AMP3Network(), PerfectCritic()
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()),
                           lr=args.lr)

    import os
    done = 0
    if args.init_from and not os.path.exists(args.ckpt):
        sd = torch.load(args.init_from)
        actor.load_state_dict(sd["actor"]); critic.load_state_dict(sd["critic"])
        opt.load_state_dict(sd["opt"]); done = sd.get("updates", 0)
        print(f"initialised from {args.init_from} at {done} updates", flush=True)
    elif os.path.exists(args.ckpt):
        sd = torch.load(args.ckpt)
        actor.load_state_dict(sd["actor"]); critic.load_state_dict(sd["critic"])
        opt.load_state_dict(sd["opt"]); done = sd.get("updates", 0)
        print(f"resumed at {done} updates", flush=True)

    t0 = time.time()
    for u in range(1, args.updates + 1):
        frac = min((done + u) / args.ent_total, 1.0)
        ent_c = args.ent_start + frac * (args.ent_end - args.ent_start)
        obs, acts, rets, perf = collect_vec(actor, game, pool, rng, args.hands)
        ent = update(actor, critic, opt, obs, acts, rets, perf, ent_c)
        if u % 100 == 0:
            print(f"  u{done+u:5d}  ret {rets.mean():+.3f}  ent {ent:.3f} "
                  f"(c={ent_c:.3f})  [{time.time()-t0:.0f}s]", flush=True)

    total = done + args.updates
    torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(),
                "opt": opt.state_dict(), "updates": total}, args.ckpt)
    print(f"saved at {total} updates, {time.time()-t0:.0f}s "
          f"({args.updates*args.hands/(time.time()-t0):.0f} hands/s)", flush=True)


if __name__ == "__main__":
    main()
