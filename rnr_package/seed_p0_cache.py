"""
Seed the p = 0 cache from p = 0 runs you have already finished, so remaining p = 0 jobs
become evaluation-only (about a minute instead of 20-50).

At p = 0 with independent restriction the solve does not depend on the pair or alpha,
only on the seat. Any finished p = 0 result for a seat can serve every pair and alpha
in that seat. Validated: re-evaluating the CC p = 0 strategy against the CO pair
reproduces the from-scratch CO p = 0 result to every printed digit.

    python seed_p0_cache.py
"""
import glob
import json
import os

import numpy as np

done = set()
for f in sorted(glob.glob("results/rnr_*_independent_selfish_p0.0.json")):
    r = json.load(open(f))
    s = r["seat"]
    cache = f"results/p0cache_s{s}_selfish.npz"
    if s in done or os.path.exists(cache):
        continue
    strat = f.replace(".json", "_strategy.npy")
    if not os.path.exists(strat):
        continue
    np.savez(cache, s0=np.load(strat), gaps=np.array(r["gaps"], dtype=object), iters=r["iters"])
    done.add(s)
    print(f"seat {s}: cache seeded from {os.path.basename(f)}")
if not done:
    print("nothing seeded (no finished p=0 runs, or caches already exist)")
