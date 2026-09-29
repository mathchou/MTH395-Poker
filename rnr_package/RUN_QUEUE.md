# RNR experiments to run locally

Everything here is resumable: stop any time, rerun the same command, finished jobs are
skipped. Works in PowerShell.

## Setup (once)

```powershell
cd rnr_package
python tree_engine.py build     # ~2 min, writes leduc3p_tree.npz (138 MB)
python tree_engine.py verify    # must print matching engine/OpenSpiel numbers
```

`verify` should show the engine's EV `[-0.093603 -0.008674 0.102278]` equal to OpenSpiel's,
and matching seat-1 best-response values. If they don't match, stop and tell me.

## The queue

| priority | what | why | jobs | time, one process |
|---|---|---|---|---|
| **1** | FF, alpha 0.35, seat 0, symmetric restriction, self-interested, p in {0, .25, .5, .75, .9, .97}, plus linear baseline | **RERUN-21/22 — the headline.** Draft v5's RNR trusted seat 2's model; seat 2 can punish it for -3.6. | 7 | ~1.4 h |
| 2 | FF, one-sided, self-interested, p in {.25, .5, .75} | RERUN-22: does the free opponent's objective matter? | 3 | ~0.7 h |
| **3** | **all 16 ordered pairs** (OO, OR, ..., FF — order matters), alpha 0.35, seat 0, p in {0, .5, .9}, plus linear | RERUN-10: is the result specific to over-folders, and does opponent seat order change it? OO is a control (RNR should equal CFR) | 60 | ~10.5 h |
| **4** | **we sit in seat 1 or 2**: FF on the fine grid, plus RR, CC, FR, RF and the OO control on the broad grid | RERUN-16: seat changed earlier gains by 10-40% and reversed which opponent type pays most. Without this, every claim is seat-0-only | 52 | ~9.3 h |
| 5 | all 16 ordered pairs, alpha 0.1, seat 0 | Does the picture hold for mild bias? | 64 | ~11 h |
| 6 | remaining 11 ordered pairs, alpha 0.35, seats 1 and 2 | Full seat coverage | 80 | ~14 h |

**Pair notation is seat order.** For a learner in seat 0, `FR` means seat 1 is the
over-folder and seat 2 the over-raiser; `RF` is the reverse. Earlier measurements showed
swapping order can change the value by over 50%, so both are run.

**Pair notation depends on our seat.** The two letters are the opponents in increasing
seat order: in seat 0, `FR` = seat 1 folder, seat 2 raiser; in seat 1, `FR` = seat 0
folder, seat 2 raiser; in seat 2, `FR` = seat 0 folder, seat 1 raiser.

Priorities 1-4 are the paper; 5-6 are robustness. With 3 shards, priority 3 takes about
3.5 hours and priority 4 about 3.

```powershell
python run_queue.py --priority 1        # start here
python run_queue.py --list --priority 1 2 3 4   # see status
python summarize.py                      # tables + plots from whatever has finished
```

To use more cores, open N PowerShell windows and give each a different shard:

```powershell
python run_queue.py --priority 3 --shard 0 --nshards 3
python run_queue.py --priority 3 --shard 1 --nshards 3
python run_queue.py --priority 3 --shard 2 --nshards 3
```

Each process uses about 0.5 GB of RAM. Keep the machine from sleeping.

## What to send back

The output of `python summarize.py`, or the whole `results/` folder. Any row flagged
"not converged" means that job needs more iterations: delete its `.json` and rerun with
`ITERS` raised in `run_queue.py`.

## What each result contains

- `vs_biased` — our EV against the biased pair
- `selfish_seatK` / `adversary_seatK` — our EV when opponent K switches to a
  self-interested / adversarial best response to us, the other opponent staying biased
- `worst_selfish` / `worst_adversary` — the minimum over both opponents. **These are the
  numbers that matter.** Draft v5 only checked seat 1.
- `gaps` — convergence: how much each role could gain by deviating in the modified
  game. Below 0.005 is good.

## Already verified here

- One-sided adversarial RNR at p = 0.5 reproduces draft v5 exactly (+1.1837 vs biased,
  +0.3270 vs seat-1 adversary).
- Symmetric self-interested RNR at p = 0.5, 120 iterations (not converged, gaps ~0.015):
  +0.481 vs biased, -0.014 worst selfish punisher, -0.269 worst adversary.
