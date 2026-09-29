# REPRODUCE.md

Every command needed to regenerate the results in `README.md`, in order, with
runtimes and expected output.

**Read this first:** the numbers below came from single runs at seed 0. Seed-to-seed
variance has never been measured in this project. Expect the qualitative pattern to
hold and the third decimal place not to. Anything you plan to put in a paper needs
five seeds.

---

## 0. Setup

```bash
pip install open_spiel torch
```

`open_spiel` ships prebuilt manylinux wheels, so no compilation. Tested with
open_spiel 2.0.2 and torch 2.13. CPU only throughout; no GPU is used or helpful.

Confirm the game loads and note the deck size, because it is not what the 2-player
game would lead you to expect:

```bash
python -c "
import pyspiel
for n in [2,3]:
    g = pyspiel.load_game('leduc_poker', {'players': n})
    print(n, 'players: deck', len(g.new_initial_state().chance_outcomes()))
"
```

```
2 players: deck 6
3 players: deck 8
```

open_spiel scales the deck as 2 suits x (num_players + 1) ranks. The 3-player game
has 4 ranks, not the familiar J/Q/K.

---

## Dependency graph

Some steps must precede others. The hard constraints:

```
build_osm.py  ──> measured_features.npz ──> train.py --style measured
              └─> osm.pt               ──> train.py --style osm
                                       └─> train_fast.py --style measured

train_fast.py ──> ck_fast.pt ──> eval_full.py
                             └─> fine-tuning A/B (step 5)
```

`verify.py` and `ceiling.py` depend on nothing and can run any time.

---

## 1. Verification suite

```bash
python verify.py
```

**Runtime:** ~12 minutes. Individual checks can be run alone: `python verify.py v3`.

**Expected output:**

```
PASS  V1 all 10 archetypes execute            0 crashed (upstream: 5/37)
PASS  V2 archetypes differentiated            spread 1.6895, std 0.4829
PASS  V3 feature profiles match archetypes    4/4 ordering checks
PASS  V4 network shapes and masking           OSM 147,332 params, AMP3 175,076 params
PASS  V5 OSM learns style from history        57.3% MSE reduction vs baseline
PASS  V6 open_spiel integration               nash_conv 12.5918
```

V5 imports `osm_budget.run()` and trains a network, so it is the slowest check.
V6 tabularizes a randomly initialised network and runs `nash_conv`; its value is
arbitrary and only needs to be positive and finite.

If V1 reports crashes, the archetype parameters in `styles.py` have been edited.
If V3 shows VPIP near 1.0 for every archetype, the free-check correction in
`StyleTracker.observe_action` has been lost.

---

## 2. Reference ceilings

**Run this before interpreting any agent result.** It is the single most important
step in the file.

```bash
python ceiling.py
```

**Runtime:** ~4 minutes.

**Expected output:**

```
=== seat 0 vs two REGULAR ===
  BEST POSSIBLE (exact best response): +1.1679
  always fold                        : -0.6150
  scripted ROCK                      : +0.1876
  scripted CONSERVATIVE              : +0.1529
  scripted REGULAR                   : -0.0052

=== seat 0 vs two MANIAC ===
  BEST POSSIBLE (exact best response): +3.2633
  always fold                        : -1.9082
  scripted ROCK                      : +1.2352
  scripted CONSERVATIVE              : +1.1752
  scripted REGULAR                   : +1.2584
```

These four numbers per opponent are the axis for every later plot: the ceiling, the
always-fold floor, and two scripted baselines. An agent that does not beat ROCK has
not learned to play, and comparisons among such agents are meaningless.

---

## 3. Build the OSM artifacts

```bash
python build_osm.py
```

**Runtime:** ~4 minutes. Writes `measured_features.npz` and `osm.pt`.

**Expected output (last two lines):**

```
  LAG                0.664 0.338 0.277 0.964
OSM test MSE 0.00717 vs baseline 0.02013 (64.4% reduction), n=4,990
```

The printed table is the measured (vpip, pfr, afq, wtsd) profile of each archetype.
Note WTSD spans only 0.87–0.97 across all ten, so the style vector is effectively
three-dimensional. This is a known weakness, not a bug.

Optional, the preliminary Experiment C sweep:

```bash
python osm_budget.py 1 5
```

**Runtime:** ~4 minutes.

```
hands pooled  samples   test MSE   baseline  reduction
           1     5990    0.01327    0.01977      32.9%
           5     5990    0.00749    0.01933      61.3%
```

The script scales `hands_per_style` with `K` so sample count stays constant. **Do
not remove that.** Without it, pooling more hands appears to make predictions worse,
which is a training-set-size artifact and not a finding.

---

## 4. Main training run

```bash
python train_fast.py --updates 4700 --hands 256 --ckpt ck_fast.pt
```

**Runtime:** 30–45 minutes. Throughput starts near 1,350 hands/s and falls to about
360 hands/s as the policy stops folding and hands run longer.

The script resumes from `--ckpt` if the file exists, so you can run it in pieces:

```bash
python train_fast.py --updates 1200 --hands 256 --ckpt ck_fast.pt   # repeat
```

Save intermediate checkpoints if you want the training-trajectory story:

```bash
cp ck_fast.pt ck_4400.pt    # after reaching 4,400
```

**Progress lines look like:**

```
  u 1200  ret +0.478  ent 0.130 (c=0.072)  [228s]
  u 2500  ret +0.361  ent 0.064 (c=0.062)  [274s]
  u 4400  ret +1.305  ent 0.019 (c=0.049)  [216s]
```

`ret` is mean return against the training pool, `ent` is policy entropy, `c` is the
annealed entropy coefficient.

### Evaluate

```bash
python eval_full.py ck_fast.pt
```

**Runtime:** ~2 minutes (23 s tabularize, ~30 s per exact EV, ~52 s nash_conv).

**Expected output at 4,700 updates:**

```
checkpoint ck_fast.pt  updates=4700
  mean policy entropy over 25,800 infosets: 0.0001 (max 1.099)
  EV vs 2x REGULAR   +0.3291
  EV vs 2x MANIAC    +1.5936
  nash_conv 7.1482 (51s)
```

Compare against step 2: the agent beats scripted ROCK on both opponents, capturing
28% of available best-response value against REGULAR and 49% against MANIAC. Mean
policy entropy of 0.0001 against a maximum of 1.099 means the policy is effectively
pure.

For a faster EV-only check during long runs:

```bash
python eval_fast.py ck_fast.pt      # ~1 minute, no nash_conv
```

---

## 5. Conditioning A/B (fine-tuning design)

Fork the converged checkpoint and fine-tune two copies for a matched budget. Common
initialisation controls for base skill, so the only difference over the final 64,000
hands is the style input.

Requires `measured_features.npz` from step 3.

```bash
cp ck_fast.pt ck_ft_none.pt
cp ck_fast.pt ck_ft_meas.pt

python train_fast.py --updates 250 --hands 256 --style none     --ckpt ck_ft_none.pt
python train_fast.py --updates 250 --hands 256 --style measured --ckpt ck_ft_meas.pt

python eval_full.py ck_ft_none.pt none
python eval_full.py ck_ft_meas.pt measured
```

**Runtime:** ~4 minutes each for training, ~2 minutes each for evaluation.

**Expected output:**

| | Unconditioned | Style-conditioned |
|---|---|---|
| EV vs two REGULAR | +0.3132 | +0.3008 |
| EV vs two MANIAC | +1.3028 | +1.6742 |
| nash_conv | 6.7912 | 8.0388 |
| Mean policy entropy | 0.0001 | 0.0057 |

The second argument to `eval_full.py` must match the training mode. Evaluating a
style-conditioned checkpoint with `none` feeds it a zero vector it was not trained
on and the numbers will be wrong.

**Known limitation of this design:** the shared initialisation was trained with a
zero style input, so the style encoder branch had learned nothing. Feeding it real
features mid-run is a distribution shift. Training both from scratch at full budget
is the cleaner experiment and has not been run.

---

## 6. The undertrained comparison (for the record only)

These results are in `README.md` and are **void** — every agent sits near the
always-fold floor and the differences between them are noise. Reproduce them only if
you want to see the failure mode.

```bash
python train.py --style none      --updates 680 --hands 128 --ent 0.08 --ckpt ck_none.pt
python train.py --style permuted  --updates 680 --hands 128 --ent 0.08 --ckpt ck_perm.pt
python train.py --style osm       --updates 680 --hands 128 --ent 0.08 --ckpt ck_osm.pt
python train.py --style measured  --updates 680 --hands 128 --ent 0.08 --ckpt ck_meas.pt

python compare.py unconditioned
python compare.py permuted
python compare.py osm
python compare.py measured
```

**Runtime:** ~10 minutes per agent with `train.py` (the unvectorised trainer, ~190
hands/s), plus ~3 minutes per `compare.py` call.

`train.py --style osm` requires `osm.pt`; `--style measured` requires
`measured_features.npz`. Both come from step 3.

Use `train_fast.py` for anything new. `train.py` is kept because it is the only
trainer supporting `permuted` and `osm` modes, which have not been ported.

---

## Troubleshooting

**`ValueError: Game must be a 2-player game`** — you called
`exploitability.exploitability()`. Use `exploitability.nash_conv()`.

**`IndexError: index 3 is out of bounds`** in a one-hot encoder — something assumed
a 3-rank deck. The 3-player game has 4 ranks. See `styles.NUM_RANKS`.

**`nash_conv` takes forever** — you passed a network policy directly. Wrap it with
`tabularize.tabularize()` first. A torch forward pass at each of 1.83M tree nodes is
impractical; caching the 25,800 information states takes 23 s and turns `nash_conv`
into a 50 s operation.

**Training stalls with entropy near 0.05 early** — entropy collapse. `train.py` at
`--ent 0.01` collapses from 0.80 to 0.046 by update 150 and never recovers.
`train_fast.py` anneals from 0.08 to 0.01 over `--ent-total` updates, which avoids
it. If you shorten a run, shorten `--ent-total` to match or the coefficient never
decays.

**A `train_fast.py` chunk prints progress but no `saved at N updates` line** — the
process was killed before the save. The checkpoint is unchanged and that chunk is
lost. Use fewer updates per invocation.

**Every archetype has VPIP near 1.0** — the free-check correction is missing.
Leduc round 1 opens with no outstanding bet, so `CALL` is a free check and must not
count as voluntary money. `FOLD` being legal is how you detect facing a bet.

---

## Timing summary

| Step | Runtime |
|---|---|
| 1. `verify.py` | 12 min |
| 2. `ceiling.py` | 4 min |
| 3. `build_osm.py` | 4 min |
| 4. `train_fast.py` to 4,700 | 30–45 min |
| 4. `eval_full.py` | 2 min |
| 5. Conditioning A/B | 12 min |
| **Total** | **~75–90 min** |

Step 6 adds roughly 50 minutes.

---

## What is not reproducible

- **Exact numbers.** Single seed, and torch version differences will shift results.
  The qualitative pattern should hold: the agent beats ROCK, entropy collapses to
  near zero, conditioning helps against MANIAC and not REGULAR, and conditioning
  raises `nash_conv`.
- **Checkpoints are not distributed.** Regenerate with step 4.
- **The 10,000-update run.** Training stopped at 4,950. Whether performance plateaus
  is unknown; between 4,400 and 4,700 every measured quantity was flat, but 300
  updates is too short a window to call it.
- **The CFR baseline.** Never built. `pyspiel.CFRSolver` at 4.3 s/iteration needs
  about 70 minutes for 1,000 iterations. It is the lambda=0 anchor for Experiment E
  and remains the oldest unfinished item in the project.
