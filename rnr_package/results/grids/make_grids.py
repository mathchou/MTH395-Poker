import json, glob, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
R = sys.argv[1] if len(sys.argv) > 1 else "results/"
OUT = sys.argv[2] if len(sys.argv) > 2 else "frontier_grids/"
os.makedirs(OUT, exist_ok=True)
T = "OCRF"
EQ = {s: json.load(open(f"{R}linear_OO_a0.35_s{s}.json"))["rows"][0]["vs_biased"] for s in range(3)}

def load(pair, a, s):
    pf = f"{R}pess/pess_{pair}_a{a}_s{s}.json"
    if os.path.exists(pf):
        r = json.load(open(pf))
        lin = [(v["gain"], v["worst"]) for w, v in sorted(r["linear"].items(), key=lambda x: float(x[0]))]
        return lin, {float(p): (v["gain"], v["worst"]) for p, v in r["rnr"].items()}, "pessimistic"
    lf = f"{R}linear_{pair}_a{a}_s{s}.json"
    if not os.path.exists(lf): return None
    e = EQ[s]; lin = [(x["vs_biased"] - e, x["worst_selfish"] - e) for x in json.load(open(lf))["rows"]]
    rnr = {}
    for f in glob.glob(f"{R}rnr_{pair}_a{a}_s{s}_independent_selfish_p*.json"):
        if os.path.basename(f)[:-5].split("_p")[-1].replace(".", "").isdigit():
            x = json.load(open(f)); rnr[x["p"]] = (x["vs_biased"] - e, x["worst_selfish"] - e)
    return lin, rnr, "exact"

def grid(a, s, zoom):
    opps = [o for o in range(3) if o != s]
    D = {t1 + t2: load(t1 + t2, a, s) for t1 in T for t2 in T}
    src = {d[2] for d in D.values() if d}
    allx = [w for d in D.values() if d for _, w in d[0]] + [w for d in D.values() if d for _, w in d[1].values()]
    ally = [g for d in D.values() if d for g, _ in d[0]] + [g for d in D.values() if d for g, _ in d[1].values()]
    if zoom:   # span every RNR and CFR point; best responses and linear tails fall off the left edge
        keyx = [w for d in D.values() if d for _, w in d[1].values()] + [d[0][0][1] for d in D.values() if d]
        x_lo = min(keyx) - 0.25
    else:
        x_lo = min(allx) - 0.1
    x_hi = max(allx) + 0.1
    y_lo, y_hi = min(-0.05, min(ally) - 0.05), max(ally) * 1.08 + 0.02
    fig, axs = plt.subplots(4, 4, figsize=(17, 15), sharex=True, sharey=True)
    for i, t1 in enumerate(T):
        for j, t2 in enumerate(T):
            ax = axs[i, j]; pair = t1 + t2; d = D[pair]
            ax.set_title(pair, fontsize=11, fontweight="bold")
            ax.add_patch(Rectangle((0, 0), x_hi + 1, y_hi + 1, color="tab:green", alpha=.08, zorder=0, lw=0))
            ax.axhline(0, color="k", lw=.5); ax.axvline(0, color="k", lw=.5); ax.grid(alpha=.25)
            if d is None:
                ax.text(.5, .5, "not run", ha="center", va="center", transform=ax.transAxes, color="gray"); continue
            lin, rnr, _ = d
            ax.plot([w for _, w in lin], [g for g, _ in lin], "-o", color="gray", ms=3, lw=1, clip_on=True)
            pts = sorted((p, v) for p, v in rnr.items() if p > 0)
            if pts:
                ax.plot([v[1] for _, v in pts], [v[0] for _, v in pts], "-s", color="tab:blue", ms=4, lw=1.2)
                for p, v in pts: ax.annotate(f"{p:g}", (v[1], v[0]), fontsize=7, color="tab:blue", xytext=(3, 2), textcoords="offset points")
            elif pair != "OO":
                ax.text(.02, .92, "RNR runs not finished", transform=ax.transAxes, fontsize=8, color="tab:orange")
            if 0.0 in rnr: ax.plot(rnr[0.0][1], rnr[0.0][0], "s", mfc="none", mec="tab:blue", ms=6)
            ax.plot(lin[0][1], lin[0][0], "*", color="black", ms=11, zorder=5)
            bg, bw = lin[-1]
            if bw >= x_lo:
                ax.plot(bw, bg, "D", color="tab:red", ms=5, zorder=5)
            else:  # pinned at the edge, labelled with its true value
                ax.plot(x_lo, bg, "<", color="tab:red", ms=7, zorder=5, clip_on=False)
                ax.annotate(f"BR {bw:+.2f}", (x_lo, bg), fontsize=7, color="tab:red", xytext=(6, -3), textcoords="offset points")
        axs[i, 0].set_ylabel(f"seat {opps[0]} = {t1}\n\ngain vs equilibrium", fontsize=9)
    for j, t2 in enumerate(T): axs[3, j].set_xlabel(f"worst case vs equilibrium\n\nseat {opps[1]} = {t2}", fontsize=9)
    axs[0, 0].set_xlim(x_lo, x_hi); axs[0, 0].set_ylim(y_lo, y_hi)
    h = [Line2D([], [], color="gray", marker="o", ms=3, lw=1, label="linear mixing"),
         Line2D([], [], color="tab:blue", marker="s", ms=4, lw=1.2, label="RNR (label = p)"),
         Line2D([], [], color="none", marker="s", mfc="none", mec="tab:blue", ms=6, label="RNR p = 0"),
         Line2D([], [], color="none", marker="*", mfc="black", mec="black", ms=11, label="CFR"),
         Line2D([], [], color="none", marker="D", mfc="tab:red", mec="tab:red", ms=5, label="best response")]
    fig.legend(handles=h, loc="upper right", fontsize=9, ncol=5, bbox_to_anchor=(0.98, 0.995))
    view = "zoomed: best responses beyond the left edge are pinned there with their true value" if zoom else "full range"
    fig.suptitle(f"alpha = {a}, we sit in seat {s}   (rows: seat {opps[0]}'s type, columns: seat {opps[1]}'s type)   "
                 f"worst case: {'/'.join(sorted(src))} punisher", fontsize=13, x=0.02, ha="left", y=0.995)
    fig.text(0.5, 0.003, f"All panels share the same axes ({view}). Each marker is one strategy, evaluated exactly. "
             "Green: better than equilibrium on both axes.", ha="center", fontsize=9, style="italic")
    plt.tight_layout(rect=(0, 0.01, 1, 0.975))
    f = f"{OUT}frontier_grid_alpha{a}_seat{s}_{'zoom' if zoom else 'full'}.png"
    plt.savefig(f, dpi=110); plt.close(); return f

if __name__ == "__main__":
    for a, seats in ((0.35, (0, 1, 2)), (0.2, (0,)), (0.1, (0,))):
        for s in seats:
            for z in (False, True):
                print(os.path.basename(grid(a, s, z)))
