"""EV + exploitability + mean policy entropy for a checkpoint."""
import sys, time, torch, numpy as np, pyspiel
import encode
from styles import make_game, ArchetypePolicy, PlayerType
from networks import AMP3Network, FULL_STYLE_DIM
from tabularize import tabularize
from evaluate import expected_returns_exact, MixedProfile, nash_conv_3p

ck = sys.argv[1]
game = make_game()
sd = torch.load(ck)
net = AMP3Network(); net.load_state_dict(sd["actor"]); net.eval()
STYLE = sys.argv[2] if len(sys.argv) > 2 else "none"
if STYLE == "measured":
    from train import measured_style_vector
    _cur = [None]
    def Zfor():
        return torch.tensor(measured_style_vector([_cur[0], _cur[0]])).unsqueeze(0)
else:
    def Zfor():
        return torch.zeros(1, FULL_STYLE_DIM)

class P(pyspiel.Policy):
    def action_probabilities(self, st, pid=None):
        if pid is None: pid = st.current_player()
        legal = st.legal_actions(pid)
        m = torch.zeros(1, encode.NUM_ACTIONS, dtype=torch.bool)
        for a in legal: m[0, a] = True
        with torch.no_grad():
            lg, _ = net(torch.tensor(encode.encode_personal(st, pid)).unsqueeze(0),
                        torch.tensor(encode.encode_public(st)).unsqueeze(0),
                        torch.tensor(encode.encode_position(pid)).unsqueeze(0),
                        torch.tensor(encode.encode_action_history([], pid)).unsqueeze(0),
                        Zfor(), legal_mask=m)
            p = torch.softmax(lg, -1)[0]
        return {a: float(p[a]) for a in legal}

if STYLE == "measured":
    _cur[0] = ArchetypePolicy(PlayerType.REGULAR)
tab = tabularize(game, P(), verbose=False)
ents = []
for probs in tab.table.values():
    v = np.array([p for p in probs.values() if p > 0])
    ents.append(float(-(v * np.log(v)).sum()))
print(f"checkpoint {ck}  updates={sd.get('updates','?')}")
print(f"  mean policy entropy over {len(ents):,} infosets: {np.mean(ents):.4f} (max {np.log(3):.3f})")
for t in [PlayerType.REGULAR, PlayerType.MANIAC]:
    opp = ArchetypePolicy(t)
    print(f"  EV vs 2x {t.name:<9} {expected_returns_exact(game, [tab, opp, opp])[0]:+.4f}", flush=True)
t0 = time.time()
print(f"  nash_conv {nash_conv_3p(game, MixedProfile([tab, tab, tab])):.4f} ({time.time()-t0:.0f}s)")
