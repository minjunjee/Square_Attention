#!/usr/bin/env python3
"""Presentation figures, from the result files.

usage: python3 make_figures.py      -> results/figures/fig_depth_curves.png, results/figures/fig_speed.png
"""
import json
from collections import defaultdict

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

from make_fair import AQUA, BAND, BLUE, GRAY, INK, INK2, MUTED, ORANGE, SURFACE, YELLOW, line, panel  # noqa: F401

if not any("Inter" in f.name for f in font_manager.fontManager.ttflist):
    plt.rcParams["font.family"] = ["DejaVu Sans"]

RUNS = [json.loads(l) for l in open("results/res_server.jsonl")]


def ok(r):                                                           # same rules as make_presentation.py
    return all(v >= 0.99 for v in r["acc_in_dist"].values())


def ind(r):
    return float(np.mean(list(r["acc_in_dist"].values())))


def chosen(pred):
    by = defaultdict(list)
    for r in RUNS:
        if pred(r):
            by[r["lr"]].append(r)
    lr = max(by, key=lambda x: np.mean([ind(r) for r in by[x]]))
    return lr, by[lr]


def size(n):
    return f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.0f}K"


def depth_curves(out):
    at = lambda d: lambda r: r["acc_deep"][str(d)]
    looped = lambda d: lambda r: r["acc_deep_by_loops"]["16"][str(d)]          # the looped model at its best test-time count
    series = [(lambda r: f"Squared read (proposed), r = {r['r_eval']}", BLUE, lambda r: r["tag"] == "e2fix", at, 6),
              (lambda r: "DeltaNet, 2 layers", ORANGE, lambda r: r["tag"] == "e2" and r["model"] == "delta" and r["L"] == 2, at, 4),
              (lambda r: "Transformer, 12 layers", AQUA, lambda r: r["tag"] == "e2" and r["model"] == "tf" and r["L"] == 12, at, 4),
              (lambda r: f"DeltaNet block, trained with {r['loops']} passes, 16 at test", YELLOW,
               lambda r: r["tag"] == "e3" and r["model"] == "delta", looped, 4)]
    fig, ax = plt.subplots(figsize=(8.6, 5.0), dpi=200)
    panel(ax, "100,000 steps on depths 1 to 4. Mean over seeds that reach 99% on all training depths, at each model's chosen learning rate", 4)
    xs = list(range(1, 9))
    lr, rs = chosen(series[0][2])
    guess = [100 * np.mean([r["guess_deep"][str(d)] for r in rs]) for d in xs]
    ax.plot(xs, guess, color=GRAY, linewidth=2, solid_capstyle="round", zorder=2, label="Guessing among chain-end digits")
    for name, color, pred, field, z in series:
        lr, rs = chosen(pred)
        good = [r for r in rs if ok(r)]
        ys = [100 * np.mean([field(d)(r) for r in good]) for d in xs]
        line(ax, xs, ys, color, f"{name(good[0])}, {size(good[0]['params'])} params ({len(good)} of {len(rs)} seeds)", z=z)
        print(f"{name(good[0])}: lr {lr:g}, depth 8 = {ys[-1]:.1f}%")
    ax.set_xticks(xs)
    ax.set_ylabel("Accuracy")
    ax.text(0.62, 104.5, "depths seen in training", color=MUTED, fontsize=9.5, va="bottom")
    ax.legend(loc="upper left", bbox_to_anchor=(-0.01, -0.16), fontsize=9.5, ncol=1, handlelength=2.2, labelcolor=INK2, borderaxespad=0)
    fig.suptitle("Following variable chains deeper than in training", x=0.012, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def speed(out):
    best = {}
    for l in open("results/cost_best_long.jsonl"):
        r = json.loads(l)
        best[r["T"]] = r
    Ts = sorted(best)

    def fastest(r, keys):
        return min(r[k] for k in keys if isinstance(r.get(k), (int, float)))

    per_k = lambda ms, r: 1000 * ms / r["tokens"]                    # ms per 1,000 tokens
    att = [per_k(fastest(best[T], ("sdpa_flash", "sdpa_cudnn", "sdpa_efficient")), best[T]) for T in Ts]
    dn = [per_k(fastest(best[T], ("delta_chunk", "delta_fused_recurrent")), best[T]) for T in Ts]
    ours = [per_k(best[T]["ours_layer_bf16"], best[T]) for T in Ts]
    fig, ax = plt.subplots(figsize=(8.6, 5.0), dpi=200)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    line(ax, Ts, att, AQUA, "Softmax attention, fastest of FlashAttention, cuDNN, memory-efficient")
    line(ax, Ts, ours, BLUE, "Squared-read layer (proposed): fla write kernel + fused Triton read")
    line(ax, Ts, dn, ORANGE, "DeltaNet, fla chunk kernel")
    for xs_, ys_, txt in ((Ts, att, "attention"), (Ts, ours, "proposed"), (Ts, dn, "DeltaNet")):
        ax.annotate(f"{txt}: {ys_[-1]:.3g} ms", (xs_[-1], ys_[-1]), xytext=(8, 0), textcoords="offset points", va="center", fontsize=9.5, color=INK2)
    print(f"1M tokens, ms per 1,000 tokens: attention {att[-1]:.3g}, proposed {ours[-1]:.3g}, DeltaNet {dn[-1]:.3g}; "
          f"attention / proposed {att[-1] / ours[-1]:.1f}x, proposed / DeltaNet {ours[-1] / dn[-1]:.2f}x")
    ax.set_xticks(Ts, [f"{T // 1024}K" if T < 2 ** 20 else "1M" for T in Ts])
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax.set_xlabel("Sequence length (tokens)")
    ax.set_ylabel("ms per 1,000 tokens, log scale")
    ax.set_title("Forward pass of one layer, 4 heads, d = 64, r = 5 squarings, bf16, RTX PRO 6000 Blackwell Server Edition",
                 loc="left", fontsize=10.5, color=INK2, pad=8)
    ax.legend(loc="upper left", bbox_to_anchor=(-0.01, -0.16), fontsize=9.5, handlelength=2.2, labelcolor=INK2, borderaxespad=0)
    ax.set_xlim(Ts[0] / 1.3, Ts[-1] * 3.5)
    fig.suptitle("Cost per token stays flat for the proposed layer and grows with length for attention", x=0.012, ha="left",
                 fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    depth_curves("results/figures/fig_depth_curves.png")
    speed("results/figures/fig_speed.png")
    print("figures written")
