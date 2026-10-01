"""
Which CFR table every script reads — one rule, in one place.

    from cfr_path import find_cfr
    path = find_cfr()            # or find_cfr(args.cfr) to honour an explicit --cfr

Rule: an explicit path wins; otherwise cfr_3p_5000.pkl, else cfr_3p_1000.pkl.

The 5,000-iteration table has at times been saved under the old name cfr_3p_1000.pkl.
If BOTH files exist and hold different tables, results computed by different scripts
could silently use different baselines, so this warns loudly.
"""

import os
import pickle

CANDIDATES = ("cfr_3p_5000.pkl", "cfr_3p_1000.pkl")
_checked = False


def _describe(path):
    blob = pickle.load(open(path, "rb"))
    return blob.get("iters"), blob.get("nash_conv")


def find_cfr(explicit=None):
    global _checked
    if explicit:
        if not os.path.exists(explicit):
            raise SystemExit(f"CFR table not found: {explicit}")
        return explicit
    present = [f for f in CANDIDATES if os.path.exists(f)]
    if not present:
        raise SystemExit(f"no CFR table found; looked for {', '.join(CANDIDATES)}")
    if len(present) == 2 and not _checked:
        _checked = True
        a, b = _describe(present[0]), _describe(present[1])
        if a != b:
            print(f"WARNING: {present[0]} ({a[0]} iterations) and {present[1]} ({b[0]} iterations) "
                  f"hold DIFFERENT tables. Using {present[0]}. Rename or remove one to avoid mixing "
                  f"baselines.", flush=True)
    return present[0]
