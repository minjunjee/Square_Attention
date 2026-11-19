#!/usr/bin/env python3
"""Figure and tables for the fair test (poc_fair.py). Reads res_fair.jsonl."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE, INK, INK2, MUTED, GRID, AXIS, BAND = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#f0efec"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#898781"

plt.rcParams.update({
    "font.family": ["Inter", "DejaVu Sans"], "font.size": 11, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS, "axes.linewidth": 1.0, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1.0,
    "grid.linestyle": "-", "axes.axisbelow": True, "legend.frameon": False,
})


def load(path="res_fair.jsonl"):
    return [json.loads(line) for line in open(path)] if os.path.exists(path) else []


def pct(x):
    x = 100 * x
    if x >= 99.995:
        return "100"
    return f"{x:.2f}" if x >= 99.95 else f"{x:.1f}"


def groups(rows):
    rows = [r for r in rows if r["tag"] == "fair"]
    return dict(
        ours=sorted([r for r in rows if r["model"] == "tied" and r["read"] == "square"], key=lambda r: r["seed"]),
        once=sorted([r for r in rows if r["model"] == "tied" and r["read"] == "once"], key=lambda r: r["seed"]),
        d2=[r for r in rows if r["model"] == "delta" and r["L"] == 2],
        d4=[r for r in rows if r["model"] == "delta" and r["L"] == 4],
        tf=[r for r in rows if r["model"] == "tf"],
    )


def series(r, field="acc_deep"):
    xs = sorted(int(k) for k in r[field])
    return xs, [100 * r[field][str(x)] for x in xs]


def panel(ax, title, D_train):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    ax.grid(axis="x", visible=False)
    ax.axvspan(0.5, D_train + 0.5, color=BAND, zorder=0, linewidth=0)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0", "25", "50", "75", "100%"])
    ax.set_xlabel("Chain depth of the query (hops)")
    ax.set_title(title, loc="left", fontsize=10.5, color=INK2, pad=8)


def line(ax, xs, ys, color, label, lw=2, ms=8, alpha=1.0, z=3):
    ax.plot(xs, ys, color=color, linewidth=lw, solid_capstyle="round", solid_joinstyle="round", marker="o",
            markersize=ms, markeredgecolor=SURFACE, markeredgewidth=2 if ms >= 8 else 1, label=label, alpha=alpha, zorder=z)


def figure(g, out="fig3_fair.png"):
    if not g["ours"]:
        return False
    D_train = g["ours"][0]["D_train"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), dpi=200, sharey=True)
    xs, guess = series(g["ours"][0], "guess_deep")
    # left: every seed of the proposed layer
    ax = axes[0]
    panel(ax, f"Squared read, one layer, {g['ours'][0]['params'] / 1000:.0f}K parameters: each of {len(g['ours'])} seeds", D_train)
    ax.plot(xs, guess, color=GRAY, linewidth=2, solid_capstyle="round", label="Guessing among the digits that end a chain", zorder=2)
    for i, r in enumerate(g["ours"]):
        x, y = series(r)
        line(ax, x, y, BLUE, "Squared read (ours), one line per seed" if i == 0 else None, lw=1.6, ms=6, z=4)
    ax.set_ylabel("Accuracy")
    ax.text(0.62, 104.5, "depths seen in training", color=MUTED, fontsize=9.5, va="bottom")
    ax.set_xticks(xs)
    ax.legend(loc="upper left", bbox_to_anchor=(-0.01, -0.2), fontsize=9.5, handlelength=1.6, labelcolor=INK2, borderaxespad=0)
    # right: same training budget, other models
    ax = axes[1]
    panel(ax, "Same data and number of training steps: other models", D_train)
    ax.plot(xs, guess, color=GRAY, linewidth=2, solid_capstyle="round", label="Guessing", zorder=2)
    for key, color, name in (("d2", YELLOW, "DeltaNet, 2 layers"), ("d4", MAGENTA, "DeltaNet, 4 layers"),
                             ("tf", ORANGE, "Transformer, 4 layers"), ("once", AQUA, "Our layer, state read once")):
        if g[key]:
            x, y = series(g[key][0])
            if len(g[key]) > 1:
                y = list(np.mean([series(r)[1] for r in g[key]], axis=0))
            n = f", mean of {len(g[key])} seeds" if len(g[key]) > 1 else ""
            line(ax, x, y, color, f"{name}, {g[key][0]['params'] / 1000:.0f}K parameters{n}")
    ax.set_xticks(xs)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="upper left", bbox_to_anchor=(-0.01, -0.2), fontsize=9.5, handlelength=1.6,
              labelcolor=INK2, borderaxespad=0, ncol=2)
    fig.suptitle("Variable binding task of Wu et al. (ICML 2025), raw tokens: trained on depth 1 to 4, tested on two competing chains",
                 x=0.012, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return True


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join([" --- "] + [" ---: "] * (len(header) - 1)) + "|"]
    return "\n".join(out + ["| " + " | ".join(r) + " |" for r in rows])


def tables(g):
    parts = []
    if not g["ours"]:
        return parts
    r0 = g["ours"][0]
    deep_d = sorted(int(k) for k in r0["acc_deep"])
    ind_d = sorted(int(k) for k in r0["acc_in_dist"])

    def row(name, r):
        return ([name, f"{r['params']:,}", f"{r['train_seconds']:.0f}"] + [pct(r["acc_in_dist"][str(d)]) for d in ind_d]
                + [pct(r["acc_deep"][str(d)]) for d in deep_d])

    head = ["모델", "parameter 수", "학습 시간, 초"] + [f"학습 분포 깊이 {d}" for d in ind_d] + [f"두 체인 깊이 {d}" for d in deep_d]
    rows = [row(f"제곱해서 읽는 layer 1개, seed {r['seed']}", r) for r in g["ours"]]
    rows += [row(f"같은 layer, 한 번만 읽음, seed {r['seed']}", r) for r in g["once"]]
    rows += [row("DeltaNet 2층", r) for r in g["d2"]] + [row("DeltaNet 4층", r) for r in g["d4"]]
    rows += [row("Transformer 4층", r) for r in g["tf"]]
    rows.append(["찍기 수준", "", ""] + ["" for _ in ind_d] + [pct(r0["guess_deep"][str(d)]) for d in deep_d])
    steps = r0["steps"]
    parts.append(f"**표 8. 공정한 조건의 실험. 모든 모델이 같은 프로그램으로 {steps:,} step, 한 번에 64개씩 학습했다. 숫자는 정확도 %이다.**\n\n"
                 + table(head, rows))
    best = max(g["ours"], key=lambda r: np.mean(list(r["acc_deep"].values())))
    if best.get("acc_deep_by_squarings"):
        head = ["제곱 횟수", "곱하는 횟수로 환산"] + [f"깊이 {d}" for d in deep_d]
        rows = [[str(rr), str(2 ** int(rr))] + [pct(best["acc_deep_by_squarings"][rr][str(d)]) for d in deep_d]
                for rr in sorted(best["acc_deep_by_squarings"], key=int)]
        parts.append(f"**표 9. seed {best['seed']} 모델을 시험할 때 제곱 횟수만 바꿨을 때의 정확도 %, 두 체인 시험.**\n\n" + table(head, rows))
    return parts


if __name__ == "__main__":
    g = groups(load())
    print({k: len(v) for k, v in g.items()})
    if figure(g):
        print("fig3_fair.png written")
    text = "\n\n".join(tables(g)) + "\n"
    open("fair_tables.md", "w").write(text)
    print(text)
