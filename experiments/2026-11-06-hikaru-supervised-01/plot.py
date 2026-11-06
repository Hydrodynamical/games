"""Recreate the supervised human-game experiment figure from portable logs.

Run from any directory with Matplotlib available:
    MPLCONFIGDIR=/tmp/chess-training-matplotlib python3 path/to/plot.py
Training weights and the original downloaded games are not needed.
"""
from collections import deque
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np


HERE = Path(__file__).resolve().parent
INK, MUTED, GRID = "#243642", "#596A76", "#E4EAED"
BLUE, ORANGE, LIGHT_BLUE, GREY = "#24799F", "#C57928", "#83B7CB", "#94A3AB"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.edgecolor": GRID, "svg.fonttype": "none"})


def read(name):
    return json.loads((HERE / name).read_text())


def style(ax, title):
    ax.set_title(title, loc="left", fontsize=12.5, fontweight="bold", pad=18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0, pad=7)
    ax.grid(axis="y", color=GRID)


def weighted_rolling(rows, window=100):
    """Average policy CE by Hikaru-example count within the last N updates."""
    recent = deque()
    weighted_sum = count = 0
    values = []
    for row in rows:
        n = row["policy_sample_count"]
        contribution = (row["policy_loss"] or 0) * n
        recent.append((contribution, n))
        weighted_sum += contribution
        count += n
        if len(recent) > window:
            old_sum, old_n = recent.popleft()
            weighted_sum -= old_sum
            count -= old_n
        values.append(weighted_sum / count if count else np.nan)
    return values


def main():
    config, summary, validation = read("config.json"), read("summary.json"), read("validation.json")
    initial, final = read("eval-initial/summary.json"), read("eval-final/summary.json")
    if any(record.get("status") != "completed" for record in (summary, initial, final)):
        raise ValueError("Figure requires completed training and both completed evaluations")
    updates = [json.loads(line) for line in (HERE / "updates.jsonl").read_text().splitlines() if line.strip()]
    if not updates or len(validation) < 2:
        raise ValueError("Figure requires training updates and at least two validation checkpoints")
    for key in ("split", "max_positions", "seed", "sims", "dataset_manifest_sha256"):
        if initial["config"][key] != final["config"][key]:
            raise ValueError(f"Evaluation settings differ: {key}")
    if initial["sampling"]["sampled_positions"] != final["sampling"]["sampled_positions"]:
        raise ValueError("Evaluations must use the same held-out sample size")
    first, last = validation[0], validation[-1]
    epochs = summary["completed_epoch"] - config.get("start_epoch", 0)
    val_positions = last["validation"]["value_sample_count"]
    heldout_count = final["sampling"]["sampled_positions"]

    fig = plt.figure(figsize=(15, 11.6), facecolor="#FAFCFD")
    grid = fig.add_gridspec(2, 2, left=.115, right=.96, top=.79, bottom=.18,
                           hspace=.66, wspace=.28)
    fig.text(.035, .952, "Learning from Hikaru: policy alone and MCTS", fontsize=23, fontweight="bold")
    dataset_path = HERE / "dataset_manifest.json"
    if dataset_path.exists():
        dataset = json.loads(dataset_path.read_text())
        splits = dataset["splits"]
        subtitle = (f"{dataset['accepted_games']:,} recorded games · "
                    f"{splits['train']['games']:,} training / {splits['validation']['games']:,} validation / "
                    f"{splits['test']['games']:,} test · chronological splits")
    else:
        subtitle = "Chronological whole-game splits · imitate Hikaru's moves · learn observed outcomes from both sides"
    fig.text(.035, .913, subtitle, fontsize=11.5, color=MUTED)
    fig.text(.035, .878,
             f"{epochs} supervised epoch{'s' if epochs != 1 else ''} · {summary['optimizer_steps']:,} optimizer steps · "
             f"same {heldout_count} held-out positions before and after training",
             fontsize=11, color=MUTED)

    # A: both quantities use the full 4,480-action output, not legal-conditioned NLL.
    ax = fig.add_subplot(grid[0, 0])
    style(ax, "A   Fitting human moves")
    ax.plot([row["step"] for row in updates], weighted_rolling(updates),
            color=BLUE, linewidth=2, label="Training CE · rolling 100 updates")
    val_steps = [row["step"] for row in validation]
    val_losses = [row["validation"]["policy_nll_full_action"] for row in validation]
    ax.scatter(val_steps, val_losses, color=ORANGE, s=55, zorder=4, label="Validation NLL")
    for index, (step, loss) in enumerate(zip(val_steps, val_losses)):
        ax.annotate(f"{loss:.3f}", (step, loss), xytext=(8 if index == 0 else -8, 9),
                    textcoords="offset points", ha="left" if index == 0 else "right",
                    color=ORANGE, fontsize=10)
    ax.set_xlim(-.025 * max(val_steps), max(val_steps) * 1.025)
    ax.set_ylim(0, max(max(val_losses), max(weighted_rolling(updates))) * 1.16)
    ax.set_xlabel("Optimizer step", labelpad=9)
    ax.set_ylabel("Cross-entropy / negative log likelihood (nats)", labelpad=9)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.text(0, -.25, "Full 4,480-action distribution. Only Hikaru's moves supervise policy.\n"
            "Training batches change; validation games stay fixed.",
            transform=ax.transAxes, fontsize=9.3, color=MUTED, va="top", linespacing=1.5)

    # B: imitation on exactly the same legal-conditioned held-out decisions.
    ax = fig.add_subplot(grid[0, 1])
    style(ax, "B   Matching held-out human decisions")
    measurements = [initial["heldout"]["all"], initial["heldout"]["all"],
                    final["heldout"]["all"], final["heldout"]["all"]]
    prefixes = ["raw", "search", "raw", "search"]
    x = np.arange(4)
    for metric, color, offset, label in (("top1", BLUE, -.18, "Top-one match"),
                                       ("top5", LIGHT_BLUE, .18, "Top-five includes human move")):
        values = [row[f"{prefix}_{metric}"] for row, prefix in zip(measurements, prefixes)]
        ax.bar(x + offset, values, width=.32, color=color, label=label)
        for position, value in zip(x + offset, values):
            ax.text(position, value + .018, f"{value:.1%}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x, ["Initial\npolicy", "Initial\nMCTS", "Trained\npolicy", "Trained\nMCTS"])
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_ylabel(f"Agreement across {heldout_count} positions", labelpad=9)
    ax.legend(frameon=False, fontsize=8.8, loc="upper left")
    ax.text(0, -.25,
            f"Raw legal-move NLL: {initial['heldout']['all']['raw_legal_nll']:.3f} → "
            f"{final['heldout']['all']['raw_legal_nll']:.3f} (lower is better).\n"
            "More matching top moves need not mean better probability predictions.",
            transform=ax.transAxes, fontsize=9.3, color=MUTED, va="top", linespacing=1.5)

    # C: retain unresolved cutoff games as their own category, never draws.
    ax = fig.add_subplot(grid[1, 0])
    matches = final["matches"]
    style(ax, f"C   Trained model: {matches['games']} paired match games")
    values = [matches[key] for key in ("mcts_wins", "policy_wins", "actual_draws", "unresolved_cutoffs")]
    labels = ["MCTS wins", "Policy wins", "Rules-defined draws", "Unresolved cutoffs"]
    colors = [BLUE, ORANGE, GREY, "#D9E0E4"]
    bars = ax.barh(np.arange(4), values, color=colors, height=.58)
    bars[-1].set_hatch("///")
    bars[-1].set_edgecolor("#A3AFB7")
    for i, value in enumerate(values):
        ax.text(value + .15, i, str(value), va="center", fontsize=11)
    ax.set_yticks(range(4), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(matches["games"], 1) + .8)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(False, axis="y")
    ax.grid(axis="x", color=GRID)
    ax.set_xlabel("Games", labelpad=9)
    ax.text(0, -.24, "Same final checkpoint, alternating colors from fixed openings.\n"
            "Search uses more computation per move; cutoffs remain unresolved.",
            transform=ax.transAxes, fontsize=9.3, color=MUTED, va="top", linespacing=1.5)

    # D: fixed validation labels allow a fair before/after value comparison.
    ax = fig.add_subplot(grid[1, 1])
    style(ax, "D   Value predictions on fixed validation games")
    val_rows = [first["validation"], last["validation"]]
    x = np.arange(2)
    max_value = max(row[key] for row in val_rows for key in ("value_mse", "zero_value_baseline_mse"))
    for key, color, offset, label in (("value_mse", BLUE, -.19, "Network MSE"),
                                    ("zero_value_baseline_mse", GREY, .19, "Always-zero MSE")):
        values = [row[key] for row in val_rows]
        ax.bar(x + offset, values, width=.34, color=color, label=label)
        for position, value in zip(x + offset, values):
            ax.text(position, value + .025 * max_value, f"{value:.4f}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x, ["Before training", "After training"])
    ax.set_ylim(0, max_value * 1.35)
    ax.set_ylabel("Mean squared error · lower is better", labelpad=9)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.text(0, -.24, f"Same {val_positions:,} positions from both sides; observed human outcomes.\n"
            "Results include resignations and clock effects, not perfect-play labels.",
            transform=ax.transAxes, fontsize=9.3, color=MUTED, va="top", linespacing=1.5)

    fig.text(.035, .054,
             f"One small experiment · {config['hidden']} channels · {config['device'].upper()} · "
             f"{final['config']['sims']} MCTS simulations per move · no Elo estimate",
             fontsize=13, fontweight="bold")
    fig.text(.035, .026,
             "Imitation and playing strength are separate objectives. All human test positions and match games are evaluation-only.",
             fontsize=10, color=MUTED)
    for extension in ("png", "svg"):
        path = HERE / f"finalcomparison.{extension}"
        fig.savefig(path, dpi=180, facecolor=fig.get_facecolor())
        if extension == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
