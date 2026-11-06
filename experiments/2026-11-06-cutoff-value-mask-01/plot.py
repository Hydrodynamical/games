"""Recreate the cutoff-value ablation figure from saved comparison and update logs.

Run: MPLCONFIGDIR=/tmp/chess-training-matplotlib python3 plot.py
Uses portable experiment logs when present, then the raw paths in comparison.json.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
import numpy as np


HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE / "comparison.json").read_text())
BASELINE, MASKED = "#C57928", "#24799F"
INK, MUTED, GRID = "#243642", "#596A76", "#E4EAED"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.edgecolor": GRID, "svg.fonttype": "none"})


def updates(name):
    run = DATA[name]
    archive = HERE if name == "treatment" else HERE.parent / "2026-11-06-inspection-01"
    path = archive / "updates.jsonl"
    if not path.exists():
        path = Path(run["directory"]) / "updates.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def style(ax, title):
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=38)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0, pad=7)


fig = plt.figure(figsize=(15.4, 12), facecolor="#FAFCFD")
grid = fig.add_gridspec(2, 2, left=.115, right=.97, top=.80, bottom=.18,
                        width_ratios=[1, 1.06], hspace=.80, wspace=.38)
fig.text(.035, .954, "Do unfinished-game draw labels hurt learning?", fontsize=23, fontweight="bold")
fig.text(.035, .915, "12 games per run · same seed and initial weights · 64 MCTS simulations per move · 22 updates",
         fontsize=12, color=MUTED)
fig.text(.035, .882, "Legacy: cutoff positions receive value 0     |     Masked: cutoff positions train policy only",
         fontsize=11, color=MUTED)

# A: the same full policy batch is retained; the removed portion is only value supervision.
ax = fig.add_subplot(grid[0, 0])
style(ax, "A   Which sampled positions contribute to each loss?")
completed_color, cutoff_color = "#738F9D", "#DFA55A"
labels = ["Legacy · policy", "Legacy · value", "Masked · policy", "Masked · value"]
for row, (name, head) in enumerate((("baseline", "policy"), ("baseline", "value"),
                                  ("treatment", "policy"), ("treatment", "value"))):
    run = DATA[name]
    completed = run["policy_samples"] - run["cutoff_policy_samples"]
    cutoff = run["cutoff_policy_samples"] if head == "policy" else run["cutoff_value_samples"]
    excluded = run["excluded_value_samples"] if head == "value" else 0
    ax.barh(row, completed, height=.57, color=completed_color)
    ax.text(completed / 2, row, f"{completed:,}", color="white", va="center", ha="center", fontsize=10)
    if cutoff:
        ax.barh(row, cutoff, left=completed, height=.57, color=cutoff_color)
        ax.text(completed + cutoff / 2, row, f"{cutoff:,}", va="center", ha="center", fontsize=10)
    if excluded:
        ax.barh(row, excluded, left=completed, height=.57, color="#F1F4F6", edgecolor="#9AA9B2", hatch="///")
        ax.text(completed + excluded / 2, row, f"{excluded:,} excluded", va="center", ha="center", fontsize=10,
                bbox={"facecolor": "#F1F4F6", "edgecolor": "none", "pad": 2})
ax.set_yticks(range(4), labels)
ax.invert_yaxis()
ax.set_xlim(0, 1500)
ax.set_xticks([0, 400, 800, 1200])
ax.set_xlabel("Sampled rows across all updates (replay repeats included)", labelpad=9)
ax.grid(axis="x", color=GRID)
ax.legend(handles=[Patch(color=completed_color, label="Completed games"),
                   Patch(color=cutoff_color, label="Cutoff games"),
                   Patch(facecolor="#F1F4F6", edgecolor="#9AA9B2", hatch="///", label="No value loss")],
          frameon=False, loc="lower left", bbox_to_anchor=(-.01, 1.01), ncol=3,
          fontsize=8.5, handlelength=1.3, columnspacing=1.2)

# B: a shared, balanced evaluation set avoids comparing different training populations.
ax = fig.add_subplot(grid[0, 1])
style(ax, "B   Raw values still cluster near zero")
diagnostic = DATA["common_value_diagnostics"]
positions = diagnostic["positions"]
y = np.arange(len(positions))
short_labels = ["White can mate in 1", "Black is checkmated", "Black can mate in 1",
                "White is checkmated", "Bare kings · White", "Bare kings · Black"]
ax.scatter([p["expected_value"] for p in positions], y, marker="D", color=INK, s=28, label="Correct value", zorder=4)
for key, color, offset, marker, label in (("initial", "#9DA8AF", -.21, "|", "Initial"),
                                        ("baseline", BASELINE, -.06, "o", "Legacy"),
                                        ("treatment", MASKED, .13, "o", "Masked")):
    ax.scatter([p[key] for p in positions], y + offset, color=color, s=30, marker=marker, label=label, zorder=5)
ax.axvline(0, color="#B8C4CB", linestyle="--", linewidth=1)
ax.set_xlim(-1.14, 1.14)
ax.set_ylim(5.55, -.65)
ax.set_yticks(y, short_labels, fontsize=9.5)
ax.set_xticks([-1, -.5, 0, .5, 1])
ax.set_xlabel("Value for player to move  (−1 loss · 0 draw · +1 win)", fontsize=9.5, labelpad=9)
ax.grid(axis="x", color=GRID)
ax.legend(frameon=False, ncol=4, loc="lower left", bbox_to_anchor=(-.01, 1.0),
          fontsize=8.5, columnspacing=.8, handletextpad=.4)
ax.text(0, -.27,
        f"Same-position MSE: legacy {diagnostic['mse']['baseline']:.4f} · masked {diagnostic['mse']['treatment']:.4f}\n"
        f"Always-zero baseline: {diagnostic['zero_baseline_mse']:.4f}  ·  six synthetic positions, evaluation only",
        transform=ax.transAxes, fontsize=9, color=MUTED, va="top", linespacing=1.6)

# C: identical fixed positions make policy probabilities directly comparable.
ax = fig.add_subplot(grid[1, 0])
style(ax, "C   Policy confidence on fixed tactical moves")
ids = ["white_capture_queen", "black_capture_queen", "white_mate_in_one", "black_mate_in_one"]
labels = ["White takes queen", "Black takes queen", "White mates in 1", "Black mates in 1"]
for name, color, offset, label in (("baseline", BASELINE, -.18, "Legacy"), ("treatment", MASKED, .18, "Masked")):
    probes = {p["id"]: p for p in DATA[name]["last_probe"]["positions"]}
    values = [probes[probe]["policy_probability_expected"] for probe in ids]
    ax.barh(np.arange(4) + offset, values, color=color, height=.31, label=label)
    for row, value in enumerate(values):
        ax.text(value + .005, row + offset, f"{value:.1%}", va="center", color=color, fontsize=9)
ax.set_yticks(range(4), labels)
ax.invert_yaxis()
ax.set_xlim(0, .30)
ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
ax.set_xlabel("Policy probability on the expected move (before search)", labelpad=9)
ax.grid(axis="x", color=GRID)
ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(-.01, 1.0), ncol=2, fontsize=9)
ax.text(0, -.24, "MCTS still solves 2 of 4 tactical probes in both runs.", transform=ax.transAxes,
        fontsize=9.5, color=MUTED, va="top")

# D: per-run policy fit is useful context, but the post-update search targets differ.
ax = fig.add_subplot(grid[1, 1])
style(ax, "D   Both policies fit their own search targets better")
for name, color, label in (("baseline", BASELINE, "Legacy"), ("treatment", MASKED, "Masked")):
    rows = updates(name)
    ax.plot([row["step"] for row in rows], [row["policy_kl"] for row in rows],
            color=color, label=label, linewidth=2, marker="o", markersize=3)
ax.set_xlim(.5, 22.6)
ax.set_ylim(0, 6)
ax.set_xticks([1, 5, 10, 15, 22])
ax.set_ylabel("Policy KL to sampled MCTS targets", labelpad=8)
ax.set_xlabel("Optimizer update", labelpad=9)
ax.grid(color=GRID)
ax.legend(frameon=False, loc="upper right", ncol=2, fontsize=9)
ax.text(0, -.24, "Different replay and search targets after the first update;\nthis is a training-fit metric, not a shared accuracy test.",
        transform=ax.transAxes, fontsize=9.5, color=MUTED, va="top", linespacing=1.5)

fig.text(.035, .048, "One small seeded comparison: no clear strength improvement from masking alone.",
         fontsize=13, fontweight="bold")
fig.text(.035, .022, "Both runs: 8 checkmates and 4 move-limit cutoffs. Initialization, pre-update games, and first policy loss match. Training value losses use different populations.",
         fontsize=9, color=MUTED)
for extension in ("png", "svg"):
    path = HERE / f"comparison.{extension}"
    fig.savefig(path, dpi=180, facecolor=fig.get_facecolor())
    if extension == "svg":
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    print(path)
plt.close(fig)
