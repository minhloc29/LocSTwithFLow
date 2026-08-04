#!/usr/bin/env python3
"""
plot_hierarchy_gate.py — visualize the learned hierarchy gate weights vs timestep.

Reads a gate log produced during validation (gate_log.npz: arrays `gate_t` [N]
and `gate_weights` [N, 3] = columns (patch, region, slide)) and plots the three
learned weights against t in [0, 1].

Usage:
    python scripts/plot_hierarchy_gate.py PATH/TO/gate_log.npz [--out OUT.pdf]

The plot reflects whatever the gate actually learned; it is not hardcoded to the
expected Slide@early / Region@middle / Patch@late behavior.
"""
import os
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Categorical palette (validated, light mode) from the dataviz reference.
SERIES = [
    ("Patch",  "#2a78d6"),   # slot 1 blue
    ("Region", "#eb6834"),   # slot 2 orange
    ("Slide",  "#1baf7a"),   # slot 3 aqua
]
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", help="path to gate_log.npz")
    ap.add_argument("--out", default=None, help="output pdf path (default: next to log)")
    args = ap.parse_args()

    d = np.load(args.log)
    t = np.asarray(d["gate_t"], dtype=np.float64)
    w = np.asarray(d["gate_weights"], dtype=np.float64)  # [N, 3] = patch, region, slide
    if w.ndim == 1:
        w = w[None, :]
    assert w.shape[1] == 3, f"gate_weights should have 3 columns, got {w.shape}"

    order = np.argsort(t)
    t = t[order]
    w = w[order]

    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.log)),
        "gate_curve.pdf",
    )

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for col, (name, color) in enumerate(SERIES):
        ax.plot(t, w[:, col], color=color, linewidth=2, label=name)

    ax.set_xlabel("Timestep t", color=INK_SECONDARY)
    ax.set_ylabel("Gate weight (softmax alpha)", color=INK_SECONDARY)
    ax.set_title("Hierarchy fusion gate vs timestep", color=INK_PRIMARY)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK_MUTED)

    # Direct labels for the three series (identity is never color-alone).
    for col, (name, color) in enumerate(SERIES):
        i = int(np.argmax(np.abs(np.diff(w[:, col]))) if len(t) > 1 else len(t) // 2)
        i = min(i, len(t) - 1)
        ax.annotate(
            name, xy=(t[i], w[i, col]), xytext=(6, 4),
            textcoords="offset points", color=INK_PRIMARY, fontsize=10,
        )

    ax.legend(frameon=False, loc="best", labelcolor=INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(out, format="pdf", facecolor=SURFACE)
    print(f"[*] Saved gate curve -> {out}")

    # Also emit a text summary of the schedule.
    for i, (name, _) in enumerate(SERIES):
        print(f"    {name:<6} @t=0: {w[0,i]:.3f}   @t=0.5: {w[int(len(w)/2),i]:.3f}   @t=1: {w[-1,i]:.3f}")


if __name__ == "__main__":
    main()
