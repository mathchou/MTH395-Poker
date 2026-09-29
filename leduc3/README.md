# leduc3 — AMP3 adapted to 3-player Leduc Hold'em

Style taxonomy and network architecture adapted from
[enfiyeci/amp3-poker-ai](https://github.com/enfiyeci/amp3-poker-ai), a third-party
reimplementation of Shi, Guo, Liu & Fan (2025), *Adaptive multi-player poker policy
learning based on opponent style modeling*, Neural Computing and Applications 37(19),
DOI 10.1007/s00521-025-11262-x.

Note: the upstream README attributes the paper to "Yang et al." That is incorrect.

## Install

```bash
pip install open_spiel torch
python verify.py            # ~12 min for the full suite
```

## Files

| File | Role |
|---|---|
| `styles.py` | 10 archetypes matching upstream `PlayerType`, adapted to Leduc; VPIP/PFR/AFq/WTSD tracker |
| `encode.py` | personal / public / position / action-history encoders |
| `networks.py` | OSM (bi-LSTM) and AMP3 (late-fusion actor-critic), resized for Leduc |
| `evaluate.py` | exact expected-return tree walk, `LambdaMix` frontier knob, `nash_conv_3p` |
| `tabularize.py` | caches a neural policy into a tabular policy; required before exploitability |
| `osm_budget.py` | Experiment C: OSM accuracy vs number of hands pooled |
| `verify.py` | six-check verification suite |

Not written: the AMP3 training loop, the lambda sweep for Experiment E.

## Verification results

```
PASS  V1 all 10 archetypes execute            0 crashed (upstream: 5/37)
PASS  V2 archetypes differentiated            spread 1.6895, std 0.4829
PASS  V3 feature profiles match archetypes    4/4 ordering checks
PASS  V4 network shapes and masking           OSM 147,332 params, AMP3 175,076 params
PASS  V5 OSM learns style from history        57.3% MSE reduction vs baseline
PASS  V6 open_spiel integration               nash_conv 12.5918, returns sum -5.6e-17
```

## Corrections to earlier assumptions

**The 3-player Leduc deck is 8 cards, not 6.** open_spiel scales the deck as
2 suits x (num_players + 1) ranks, so the 3-player game has 4 ranks, not the
familiar J/Q/K of the 2-player game. Verified via `chance_outcomes()`. Any
hand-strength function written for the 2-player deck will silently produce
out-of-range values.

**VPIP needs redefining for Leduc.** Round 1 opens with no outstanding bet, so CALL
is a free check. Counting it as voluntary money pins every archetype's VPIP between
0.86 and 1.00 and destroys the feature. Facing a bet is detectable as FOLD being
legal. After the fix VPIP spans 0.24 to 0.89.

**Exploitability requires tabularizing first.** `nash_conv` walks 1.8M nodes; a torch
forward pass at each is impractical. Caching the 25,800 information states takes 23 s
and makes `nash_conv` a 44 s operation.

**`exploitability.exploitability()` refuses 3-player games** with
`ValueError: Game must be a 2-player game`. Use `nash_conv()`.

## Upstream defects found

`RuleBasedStrategy._preflop_action` indexes a `thresholds` dict covering 5 of the 10
`PlayerType` values, so `RuleBased_{MANIAC, ROCK, CALLING_STATION, TAG, LAG}` raise
`KeyError`. Verified: 5 of 37 strategies crash on first use.

The README claims 64 strategies; the class docstring says 64 as 49+5+5+5; the code
builds 37. An inline comment concedes the paper's 49 "are really just 7 distinct
configs."

Upstream also adds Deep CFR and NFSP heads that do not appear in the paper's
description of AMP3. Those are omitted here as confounds.

## Measured environment facts

| Quantity | 2-player | 3-player |
|---|---|---|
| Information states | 936 | 25,800 |
| Tree nodes | 9,457 | 1,831,601 |

| Operation | Time |
|---|---|
| Python `cfr.CFRSolver` iteration | 36 s |
| C++ `pyspiel.CFRSolver` iteration | 4.3 s |
| Tabularize a neural policy | 23 s |
| `nash_conv` on a tabular policy | 44-56 s |

## Preliminary result (Experiment C)

OSM style-regression accuracy against number of hands pooled per input, training-set
size held constant at 5,990 samples:

| Hands pooled | Test MSE | Baseline | Reduction |
|---|---|---|---|
| 1 | 0.01327 | 0.01977 | 32.9% |
| 5 | 0.00749 | 0.01933 | 61.3% |

Holding sample count constant matters. An earlier sweep that did not showed K=5
performing *worse* than K=1, which was a training-set-size artifact.

## Known weaknesses

- **WTSD carries almost no signal** in Leduc: every archetype falls between 0.87 and
  0.97, because with only two betting rounds nearly everyone who enters reaches
  showdown. Consider dropping it to a 3-feature style vector, or report it as a
  limitation of the scale-down.
- **DECEPTIVE is over-tuned**: `trap_rate` 0.40 drives PFR to 0.002 and AFq to 0.000,
  so it never raises at all. Retune before using it as a held-out style.
- **ROCK and CONSERVATIVE are close** (VPIP 0.35 vs 0.40). Widen the looseness gap.
- Archetype parameters were hand-set to reproduce intended VPIP/PFR orderings, not
  fitted. They are a starting point.

## Next steps

1. Retune DECEPTIVE and ROCK, then re-run `verify.py v3`.
2. Cache a CFR baseline: `pyspiel.CFRSolver` for 1,000+ iterations, pickled.
3. Write the AMP3 training loop against `networks.AMP3Network`.
4. Sweep lambda in `evaluate.LambdaMix` for Experiment E, tabularizing at each point.

## Training results (train.py, compare.py)

A2C with an asymmetric critic (actor sees the information set, critic sees all three
private cards). Monte Carlo returns over complete hands. 680 updates x 128 hands
= 87,040 hands per agent, seed 0, CPU only, roughly 190 hands/second.

Exact EV against two REGULAR opponents, measured by tabularizing then walking the
full tree:

| Cumulative updates | Unconditioned | Oracle-conditioned |
|---|---|---|
| 50 | -1.31 | - |
| 150 | -0.69 | - |
| 250 | - | -0.71 |
| 430 | -0.69 | - |
| 500 | - | -0.28 |
| 680 | **-0.20** | **-0.05** |

Both axes at 680 updates:

| Agent | vs REGULAR | vs MANIAC | nash_conv |
|---|---|---|---|
| Unconditioned | -0.1712 | -1.7813 | 22.686 |
| Oracle-conditioned | -0.0429 | -1.5610 | 21.854 |

Reference: the scripted REGULAR archetype against two copies of itself earns
-0.0084 chips/hand. Neither agent has reached parity with a hand-written rule.

### Entropy collapse

At the default entropy coefficient of 0.01, policy entropy fell from 0.80 to 0.046
by update 150 and EV plateaued at -0.75. A near-deterministic policy in an
imperfect-information game is maximally exploitable, which is the opposite of what
this project measures. Raising the coefficient to 0.08 held entropy at 0.36-0.44 and
let learning continue well past that plateau. **This is the most important
hyperparameter in the project and it was not in the plan.**

### What these numbers do and do not show

Style conditioning helps on both axes at equal budget: better EV and slightly lower
exploitability. That is the go signal for the conditioning premise.

But: one seed, no confidence intervals, neither agent beats a scripted archetype,
both were still improving when training stopped, and the "style" input is a
hand-built proxy derived from ARCHETYPE_PARAMS rather than measured or OSM-predicted
features. The oracle advantage could be an artifact of the proxy leaking archetype
identity. Do not report any of this as a result.

### Resuming

```bash
python train.py --style oracle --updates 250 --hands 128 --ent 0.08 --ckpt ck_oracle.pt
python compare.py oracle
```

Checkpoints are not included in this bundle; regenerate them with the commands above.

## Experiment A: permutation control

Third training run, identical architecture and parameter count, but the style vector
is drawn from a randomly chosen archetype pair each hand, independent of the
opponents actually seated. Same dimensionality, same marginal distribution, zero
mutual information with the opponents. If the oracle advantage came from added
network capacity rather than style information, this run would match it.

680 updates x 128 hands, seed 0, matched budget across all three:

| Agent | vs REGULAR | vs MANIAC | nash_conv |
|---|---|---|---|
| Unconditioned | -0.1712 | -1.7813 | 22.686 |
| Permuted control | -0.1100 | -1.6109 | 21.963 |
| Oracle-conditioned | -0.0429 | -1.5610 | 21.854 |

The permuted control sits with the unconditioned agent, not with the oracle. The
gain is attributable to style information rather than to capacity. This is the
control that the original paper does not report and that the upstream
reimplementation does not run.

Caveat: the permuted agent is modestly better than unconditioned on every column,
so the extra capacity is not worth exactly zero. With one seed and no error bars
that difference is not distinguishable from noise. Run seeds 1-4 before claiming
the separation is real.

## Timeline status

| Planned week | State |
|---|---|
| W1 CFR baseline | Not done. No converged CFR policy exists; only per-iteration timings |
| W2 style library | Done, three archetypes need retuning |
| W3 OSM | Architecture and learning check done; no trained model saved |
| W4 actor-critic | Done |
| W5 AMP3 conditioned training | Done for oracle style; not for OSM-predicted style |
| W6 Experiment E | Infrastructure done; no lambda sweep, no frontier plot |
| W7 Experiments B/C | C has two preliminary points; B not started |
| W8 Experiment A | Done at one seed |

The week-5 critical-path risk did not materialise. A2C with an asymmetric critic
converges in 3-player Leduc on CPU. The tabular fallback can be dropped from the
plan.

Largest remaining gap: every result above conditions on a hand-built style proxy
derived from ARCHETYPE_PARAMS, not on OSM predictions. Closing that loop is the
next step and it is the one that decides whether any of this survives.

## Closing the OSM loop (build_osm.py, train.py --style osm)

The oracle runs above condition on `true_style_vector()`, a proxy built by hand from
ARCHETYPE_PARAMS. No observer could ever produce it: it reads the opponent's
generative parameters directly. `build_osm.py` replaces it with the measured
features an observer can actually estimate, and trains an OSM network to predict
them from pooled action history (64.4% test-MSE reduction vs the mean baseline,
n=4,990).

`--style osm` then conditions the agent on live OSM predictions. Opponents are drawn
per *session* rather than per hand (default 10 hands), the session's action history
accumulates, and OSM re-estimates at the start of each hand. Evaluation uses the
freeze-and-condition protocol: OSM observes 5 fresh hands of the eval opponent, then
the estimate is frozen and the resulting stationary policy is measured exactly.

All four agents, 680 updates x 128 hands, seed 0:

| Agent | vs REGULAR | vs MANIAC | nash_conv |
|---|---|---|---|
| Unconditioned | -0.1712 | -1.7813 | 22.686 |
| Permuted control | -0.1100 | -1.6109 | 21.963 |
| OSM-conditioned | -0.1773 | -1.7581 | 22.611 |
| Oracle proxy | -0.0429 | -1.5610 | 21.854 |

### The result

**OSM conditioning does not reproduce the oracle advantage.** It lands with the
unconditioned agent on every column, and slightly behind the permutation control.
The entire oracle gain came from information that OSM cannot recover from observed
play.

This is the single most important number produced so far, and it is a negative
result for the conditioning premise as implemented here.

### Candidate explanations, in rough order of likelihood

1. **Sample starvation.** A 10-hand session yields a handful of opponent actions.
   `osm_budget.py` shows 5 pooled hands gives 61% MSE reduction, but that pools
   hands *of the same opponent in a fixed matchup*. Live sessions are shorter and
   noisier. Try session lengths of 25, 50, 100.
2. **Distribution shift.** OSM was trained on histories where the target sat at seat
   0 with two REGULAR opponents. At inference the target is at seat 1 or 2 with a
   learning agent in the mix. Retrain OSM on histories generated under the actual
   training distribution.
3. **The proxy leaks.** `true_style_vector` may encode archetype identity more
   sharply than the four measured features do, in which case the oracle number is
   inflated and the honest ceiling is lower than -0.043. Train an
   "oracle-measured" agent on the true measured features to separate this from
   explanation 1.
4. **Four features are too few.** WTSD is nearly constant across archetypes (0.87 to
   0.97), so the style vector is effectively three-dimensional.

Explanation 3 is the one to test first: it is one training run and it bounds how
much of the oracle gap is real.

### Status of the conditioning premise

Unresolved, leaning negative. The earlier "go signal" was measured against a proxy
that no deployable system could compute. Until an oracle-measured agent is trained,
the honest statement is that style conditioning helps when the style is handed to
the agent in a privileged encoding, and does not help when estimated from play.

## Oracle-measured: bounding the real ceiling

Explanation 3 from the previous section, tested. `--style measured` conditions on the
ground-truth *measured* features of the seated opponents, read from
measured_features.npz. This is exactly what a perfect OSM would output, so it upper
bounds what style estimation can deliver. Contrast the earlier oracle, which read
generative parameters no observer can see.

Five agents, 680 updates x 128 hands, seed 0:

| Agent | Style input | vs REGULAR | vs MANIAC | nash_conv |
|---|---|---|---|---|
| Unconditioned | none | -0.1712 | -1.7813 | 22.686 |
| Permuted control | random, zero info | -0.1100 | -1.6109 | 21.963 |
| OSM-conditioned | predicted from play | -0.1773 | -1.7581 | 22.611 |
| **Oracle-measured** | **true measured features** | **-0.0784** | -1.8276 | 22.602 |
| Oracle proxy | generative parameters | -0.0429 | -1.5610 | 21.854 |

### Reading

The proxy oracle was inflated. A perfect style estimator reaches -0.078, not -0.043,
so roughly 45% of the apparent oracle advantage came from the privileged encoding
rather than from style information. Explanation 3 is partly confirmed.

But the remaining gap is real and large. Oracle-measured (-0.078) clearly beats
unconditioned (-0.171) and OSM (-0.177), while using an input OSM is trained to
produce. **The information is there and recoverable in principle; OSM is not
recovering it.** That relocates the problem from the premise to the estimator, and
makes explanations 1 and 2 the live ones.

Note also that oracle-measured does *worse* than unconditioned against MANIAC
(-1.828 vs -1.781) while doing much better against REGULAR. Conditioning is not
uniformly helpful across opponents. With one seed this may be noise, but if it
survives replication it is interesting on its own: it is the exploitation-side
analogue of the tradeoff this project set out to measure.

### Exploitability

nash_conv is flat across all five agents (21.9 to 22.7, a 3.7% spread) while
exploitation varies by a factor of four. At this training budget the agents are not
yet near enough to equilibrium for the exploitability axis to discriminate. The
frontier sweep of Experiment E needs a converged CFR baseline as its lambda=0 anchor
before those numbers mean anything, and that baseline still does not exist.

### Revised priority

1. Retrain OSM on histories from the actual training distribution (explanation 2).
   Current OSM saw the target at seat 0 against two REGULAR opponents; at inference
   it sees seats 1 and 2 with a learning agent in the mix.
2. Sweep session length: 10, 25, 50, 100 (explanation 1).
3. Only then add seeds and error bars.

The gap between oracle-measured and OSM is now the central quantity of the project.

## Long training run (train_fast.py)

`train_fast.py` vectorises the rollout: every agent decision across a batch is
served by one forward pass rather than one per decision. Throughput rose from ~190
to ~1,350 hands/s, degrading to ~360 hands/s late in training as the policy stops
folding and hands run longer. That slowdown is itself a signal.

It also anneals the entropy coefficient from 0.08 toward 0.01. A fixed value cannot
be right for a whole run: 0.01 lets the policy collapse early, 0.08 prevents sharp
play late.

### The earlier agents were starved, not broken

| | 680 updates | 4,700 updates | BR ceiling | Scripted ROCK |
|---|---|---|---|---|
| EV vs two REGULAR | -0.171 | **+0.329** | +1.168 | +0.188 |
| EV vs two MANIAC | -1.756 | **+1.594** | +3.263 | +1.235 |
| nash_conv | 22.686 | **7.148** | ~0 (CFR) | - |
| Mean policy entropy | ~0.40 | **0.0001** | - | - |

87,040 hands over 25,800 information states is about ten visits per state. At 4,700
updates x 256 hands the agent has seen ~1.2M hands, and it now beats every scripted
archetype tested, capturing 28% of available best-response value against REGULAR
and 49% against MANIAC.

**Every conditioning result reported earlier in this README is void.** The
four-agent comparison, the OSM negative result, the oracle-measured ceiling and the
permutation control were all measured on agents sitting near the fold-everything
floor. They must be rerun at this budget.

### The entropy result, and why it is not what was expected

Mean policy entropy across all 25,800 information states is **0.0001** against a
maximum of 1.099. The policy is effectively pure: at almost every information state
it plays one action with probability ~1.

The obvious prediction is that a deterministic policy in an imperfect-information
game should be highly exploitable. **It is not what happened.** nash_conv fell from
22.686 to 7.148 as entropy collapsed from ~0.40 to 0.0001.

Both quantities improved together because the early agent was not "mixed" in any
useful sense; it was near-random and badly exploitable for that reason. Improving
play quality dominated the cost of becoming deterministic. But 7.148 is still very
far from the CFR baseline's near-zero, so the agent is nowhere near equilibrium and
the determinism cost has not yet bound.

This matters for Experiment E. The naive story, that adaptation buys exploitation at
the price of exploitability, does not show up as a simple monotone tradeoff along
the training trajectory. Both axes improve together until play quality saturates.
The frontier has to be traced by a knob applied to *converged* agents, not by
reading off training checkpoints.

Between 4,400 and 4,700 updates every number is flat (EV +0.332 to +0.329, nash_conv
7.361 to 7.148, entropy unchanged). Performance may be plateauing, but 300 updates
is far too short a window to call it.

### Entropy annealing as the frontier knob

The annealing schedule is now a candidate replacement for the lambda-mixing knob in
the plan. Training several agents at fixed entropy coefficients spanning 0.005 to
0.20 would produce a frontier during training rather than by post-hoc interpolation
between a converged CFR policy and an adaptive one. Worth comparing both.

### Files

| File | Role |
|---|---|
| `train_fast.py` | vectorised trainer with entropy annealing; resumes from `--ckpt` |
| `eval_fast.py` | quick EV against two archetypes |
| `eval_full.py` | EV + nash_conv + mean policy entropy for a checkpoint |
| `ceiling.py` | exact best-response value and always-fold floor for fixed opponents |

```bash
python train_fast.py --updates 300 --hands 256      # ~210s per chunk at this stage
python eval_full.py ck_fast.pt
```

## Conditioning A/B at competent-agent budget (fine-tuning design)

Rerunning the four-agent comparison from scratch at 4,700+ updates was not
affordable. Instead: fork the converged unconditioned checkpoint and fine-tune two
copies for a matched 250 updates, one still unconditioned, one conditioned on true
measured style features. Common initialisation controls for base skill, so the only
difference is the style input over the final 64,000 hands.

Both start from ck_4700.pt, both end at 4,950 updates:

| | Unconditioned | Style-conditioned |
|---|---|---|
| EV vs two REGULAR | +0.3132 | +0.3008 |
| EV vs two MANIAC | +1.3028 | **+1.6742** |
| nash_conv | **6.7912** | 8.0388 |
| Mean policy entropy | 0.0001 | 0.0057 |

### Reading

Conditioning helps where opponents are unusual and does not help where they are
typical. Against MANIAC the conditioned agent gains +0.371 chips/hand; against
REGULAR it is marginally worse. That is the pattern the method predicts and the
opposite of what the earlier undertrained comparison showed, where conditioning
helped against REGULAR and hurt against MANIAC.

**The exploitability cost is now visible.** nash_conv is 8.04 conditioned versus
6.79 unconditioned, an 18% increase, alongside the exploitation gain against
MANIAC. This is the first measurement in the project where the two axes move in
opposite directions, which is the tradeoff the whole project exists to quantify.

The conditioned agent also retains slightly more entropy (0.0057 vs 0.0001), so it
is not simply a sharper deterministic policy.

### Caveats, which are substantial

- One seed, 250 fine-tuning updates, no error bars. A 0.371 gain and an 18%
  exploitability increase from a single run is a hypothesis, not a result.
- The common initialisation was trained with a zero style input, so the style
  encoder branch had learned nothing. Feeding it real features mid-run is a
  distribution shift and the conditioned agent may still be adapting to it.
- Conditioning uses *true measured* features, not OSM predictions. This is the
  ceiling case, not the deployable one.
- Fine-tuning from a shared checkpoint is not equivalent to training from scratch.
  Both designs should be run before anything is claimed.

### What this changes

The earlier conclusion that conditioning fails, and the later conclusion that it
succeeds, were both drawn from agents near the fold-everything floor. This is the
first comparison between agents that can actually play, and it points a third way:
conditioning is opponent-dependent and it costs exploitability.

Next: repeat with 5 seeds and longer fine-tuning, then substitute OSM predictions
for the true features to see how much of the +0.371 survives estimation.
