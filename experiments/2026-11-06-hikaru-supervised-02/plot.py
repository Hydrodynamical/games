"""Plot continued supervised training and the selected checkpoint evaluation.

Run with Matplotlib available:
    MPLCONFIGDIR=/tmp/chess-training-matplotlib python3 path/to/plot.py

Portable config.json, summary.json, and validation.json are required. Evaluation
panels remain explicitly unavailable until eval-initial/summary.json (epoch 1)
and eval-final/summary.json (the selected checkpoint) are both completed.
Online losses from changing weights are never substituted for fixed-model train
evaluation. All displayed epochs are absolute, including the resumed epoch.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np


HERE = Path(__file__).resolve().parent
INK, MUTED, GRID = "#243642", "#596A76", "#E4EAED"
BLUE, ORANGE, LIGHT_BLUE, GREY = "#24799F", "#C57928", "#83B7CB", "#94A3AB"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.edgecolor": GRID, "svg.fonttype": "none"})


def read(name, optional=False):
    path = HERE / name
    if optional and not path.exists():
        return None
    return json.loads(path.read_text())


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def style(ax, title):
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0, pad=7)
    ax.grid(axis="y", color=GRID)


def unavailable(ax, message):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(.5, .5, message, transform=ax.transAxes, ha="center", va="center",
            fontsize=12, color=MUTED, linespacing=1.6)


def history_curve(ax, history, field, metric, *, color, label, linestyle="-", marker="o"):
    rows = [row for row in history if finite(row.get(field, {}).get(metric))]
    if not rows:
        return []
    values = [row[field][metric] for row in rows]
    ax.plot([row["epoch"] for row in rows], values, color=color, label=label,
            linestyle=linestyle, marker=marker, markersize=4, linewidth=2)
    return values


def selected_marker(ax, best_epoch, best_value):
    if finite(best_epoch) and finite(best_value):
        ax.axvline(best_epoch, color="#BAC6CD", linewidth=1.2, linestyle="--", zorder=1)
        ax.scatter([best_epoch], [best_value], marker="*", s=155, color=BLUE,
                   edgecolor="white", linewidth=1, zorder=6)


def verify_evaluation_pair(initial, final):
    if initial is None or final is None or any(row.get("status") != "completed" for row in (initial, final)):
        return False
    for key in ("split", "max_positions", "seed", "sims", "dataset_manifest_sha256",
                "match_pairs", "max_plies", "root_noise", "move_sampling", "search_tree_reuse"):
        if initial["config"][key] != final["config"][key]:
            raise ValueError(f"Evaluation settings differ: {key}")
    if initial["sampling"]["sampled_positions"] != final["sampling"]["sampled_positions"]:
        raise ValueError("Evaluation sample sizes differ")
    # When available, check actual rows as well as the deterministic sampler's
    # settings; report images must not silently compare different test examples.
    positions = [HERE / name / "positions.jsonl" for name in ("eval-initial", "eval-final")]
    if all(path.exists() for path in positions):
        identities = []
        for path in positions:
            identities.append([(row["game_id"], row["ply"], row["fen"], row["human_move"])
                               for row in (json.loads(line) for line in path.read_text().splitlines() if line.strip())])
        if identities[0] != identities[1]:
            raise ValueError("Held-out position identities differ")
    matches = [HERE / name / "matches.jsonl" for name in ("eval-initial", "eval-final")]
    if all(path.exists() for path in matches):
        starts = []
        for path in matches:
            starts.append([(row["opening"], row["opening_moves"], row["mcts_color"], row["policy_color"])
                           for row in (json.loads(line) for line in path.read_text().splitlines() if line.strip())])
        if starts[0] != starts[1]:
            raise ValueError("Match openings or color assignments differ")
    return True


def main():
    config, summary, history = read("config.json"), read("summary.json"), read("validation.json")
    if not history:
        raise ValueError("At least one fixed-checkpoint evaluation is required")
    history = sorted(history, key=lambda row: row["epoch"])
    initial, final = read("eval-initial/summary.json", True), read("eval-final/summary.json", True)
    has_evaluation = verify_evaluation_pair(initial, final)
    recorded = [row for row in history if finite(row.get("validation", {}).get("policy_nll_full_action"))]
    if not recorded:
        raise ValueError("No finite validation policy NLL is recorded")
    minimum = min(recorded, key=lambda row: row["validation"]["policy_nll_full_action"])
    best_epoch = summary.get("best_epoch", minimum["epoch"])
    best_nll = summary.get("best_validation_nll", minimum["validation"]["policy_nll_full_action"])
    if best_epoch != minimum["epoch"] or not math.isclose(best_nll, minimum["validation"]["policy_nll_full_action"], rel_tol=1e-8, abs_tol=1e-10):
        raise ValueError("Recorded best checkpoint is not the minimum validation policy NLL")
    start_epoch = config.get("start_epoch", history[0]["epoch"])
    last_epoch = summary.get("completed_epoch", history[-1]["epoch"])
    status = summary.get("status", "not recorded")
    completed = status in {"completed", "early_stopped"}
    best_label = f"Selected epoch {best_epoch}" if completed else f"Best so far: epoch {best_epoch}"
    status_label = (f"early stop after {summary['epochs_without_improvement']} epochs without improvement"
                    if summary.get("stopped_early") else f"status: {status}")

    fig = plt.figure(figsize=(15, 11.7), facecolor="#FAFCFD")
    grid = fig.add_gridspec(2, 2, left=.10, right=.96, top=.79, bottom=.18,
                           hspace=.68, wspace=.30)
    fig.text(.035, .954, "More training: does validation keep improving?", fontsize=23, fontweight="bold")
    dataset = read("dataset_manifest.json", True)
    if dataset:
        splits = dataset["splits"]
        subtitle = (f"Same {dataset['accepted_games']:,} Hikaru blitz games · "
                    f"{splits['train']['games']:,} training / {splits['validation']['games']:,} validation / "
                    f"{splits['test']['games']:,} test · chronological splits")
    else:
        subtitle = "Same human-game dataset and chronological splits · continued optimizer state · fixed-checkpoint evaluation"
    fig.text(.035, .914, subtitle, fontsize=11.5, color=MUTED)
    fig.text(.035, .878, f"Resumed epoch {start_epoch} → completed epoch {last_epoch} · "
             f"{best_label} by validation policy NLL ({best_nll:.3f}) · {status_label}",
             fontsize=11, color=MUTED)

    # A: both curves evaluate a fixed checkpoint over its complete split.
    ax = fig.add_subplot(grid[0, 0])
    style(ax, "A   Policy fit and generalization")
    policy_values = history_curve(ax, history, "training_evaluation", "policy_nll_full_action",
                                  color=ORANGE, label="Training split")
    policy_values += history_curve(ax, history, "validation", "policy_nll_full_action",
                                   color=BLUE, label="Validation split")
    selected_marker(ax, best_epoch, best_nll)
    ax.annotate(f"Best: {best_nll:.3f}", (best_epoch, best_nll), xytext=(9, 11),
                textcoords="offset points", fontsize=9.5, color=BLUE)
    ax.set_ylim(0, max(policy_values) * 1.18)
    ax.set_xlim(history[0]["epoch"] - .3, history[-1]["epoch"] + .5)
    ax.set_xticks(sorted({row["epoch"] for row in history}))
    ax.set_xlabel("Absolute training epoch", labelpad=9)
    ax.set_ylabel("Full-action negative log likelihood (nats)", labelpad=9)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    policy_note = "A falling train loss with worsening validation is evidence of overfitting."
    if completed and history[-1]["validation"]["policy_nll_full_action"] > best_nll:
        policy_note = f"Validation worsens after epoch {best_epoch} while training loss falls: overfitting."
    ax.text(0, -.25, "All 4,480 actions; only Hikaru's turns. Each point uses fixed weights.\n" + policy_note,
            transform=ax.transAxes, fontsize=9.2, color=MUTED, va="top", linespacing=1.5)

    # B: genuine full-split values; never reuse online losses from changing weights.
    ax = fig.add_subplot(grid[0, 1])
    style(ax, "B   Value learning on observed human outcomes")
    value_values = history_curve(ax, history, "training_evaluation", "value_mse",
                                 color=ORANGE, label="Training MSE")
    value_values += history_curve(ax, history, "validation", "value_mse", color=BLUE, label="Validation MSE")
    value_values += history_curve(ax, history, "validation", "zero_value_baseline_mse",
                                  color=GREY, label="Always zero · validation", linestyle="--", marker=None)
    chosen = next(row for row in history if row["epoch"] == best_epoch)
    selected_marker(ax, best_epoch, chosen["validation"]["value_mse"])
    ax.set_ylim(0, max(value_values) * 1.18)
    ax.set_xlim(history[0]["epoch"] - .3, history[-1]["epoch"] + .5)
    ax.set_xticks(sorted({row["epoch"] for row in history}))
    ax.set_xlabel("Absolute training epoch", labelpad=9)
    ax.set_ylabel("Mean squared error · lower is better", labelpad=9)
    ax.legend(frameon=False, fontsize=8.8, loc="lower left")
    value_note = ("Both players' positions; results include resignations and clock effects.\n"
                  "The selected epoch minimizes policy NLL, not value error.")
    if has_evaluation:
        value_note += (f"\nSelected test value MSE: {final['heldout']['all']['value_mse']:.3f}; "
                       f"always-zero baseline: {final['heldout']['all']['zero_value_baseline_mse']:.3f}.")
    ax.text(0, -.25, value_note,
            transform=ax.transAxes, fontsize=9.2, color=MUTED, va="top", linespacing=1.5)

    # C: final tests are observational after validation-based checkpoint selection.
    ax = fig.add_subplot(grid[1, 0])
    style(ax, "C   Human move agreement: policy versus search")
    if has_evaluation:
        measurements = [initial["heldout"]["all"], initial["heldout"]["all"],
                        final["heldout"]["all"], final["heldout"]["all"]]
        prefixes, x = ["raw", "search", "raw", "search"], np.arange(4)
        for metric, color, offset, label in (("top1", BLUE, -.18, "Top-one match"),
                                           ("top5", LIGHT_BLUE, .18, "Top-five includes human move")):
            values = [row[f"{prefix}_{metric}"] for row, prefix in zip(measurements, prefixes)]
            ax.bar(x + offset, values, width=.32, color=color, label=label)
            for position, value in zip(x + offset, values):
                ax.text(position, value + .018, f"{value:.1%}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x, [f"Epoch {start_epoch}\npolicy", f"Epoch {start_epoch}\nMCTS",
                         f"Epoch {best_epoch}\npolicy", f"Epoch {best_epoch}\nMCTS"])
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.set_ylabel(f"Agreement on {final['sampling']['sampled_positions']} fixed test positions", labelpad=9)
        ax.legend(frameon=False, fontsize=8.5, loc="upper left")
        ax.text(0, -.25,
                f"Policy legal-move NLL: {initial['heldout']['all']['raw_legal_nll']:.3f} → "
                f"{final['heldout']['all']['raw_legal_nll']:.3f} (lower is better).\n"
                "This uses legal moves only; panel A uses all actions. Agreement is imitation.",
                transform=ax.transAxes, fontsize=9.2, color=MUTED, va="top", linespacing=1.5)
    else:
        unavailable(ax, "Final held-out evaluation has not completed.\nCheckpoint selection uses validation only.")

    # D: paired deterministic games are a small diagnostic, not an Elo test.
    ax = fig.add_subplot(grid[1, 1])
    style(ax, "D   Same-checkpoint matches: MCTS against policy")
    if has_evaluation and initial["matches"]["games"] and final["matches"]["games"]:
        match_rows = [initial["matches"], final["matches"]]
        categories = [("mcts_wins", "MCTS wins", BLUE), ("policy_wins", "Policy wins", ORANGE),
                      ("actual_draws", "Actual draws", GREY), ("unresolved_cutoffs", "Unresolved", "#D9E0E4")]
        left = np.zeros(2)
        for key, label, color in categories:
            values = [row[key] for row in match_rows]
            bars = ax.barh(np.arange(2), values, left=left, height=.52, color=color, label=label)
            if key == "unresolved_cutoffs":
                for bar in bars:
                    bar.set_hatch("///")
                    bar.set_edgecolor("#A3AFB7")
            for y, value in enumerate(values):
                if value:
                    ax.text(left[y] + value / 2, y, str(value), ha="center", va="center", fontsize=11,
                            color="white" if key in {"mcts_wins", "policy_wins"} else INK,
                            bbox={"facecolor": color, "edgecolor": "none", "pad": 1})
            left += values
        ax.set_yticks([0, 1], [f"Epoch {start_epoch}", f"Selected {best_epoch}"])
        ax.set_ylim(1.5, -.65)
        ax.set_xlim(0, max(row["games"] for row in match_rows) + .35)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(False, axis="y")
        ax.grid(axis="x", color=GRID)
        ax.set_xlabel("Games from fixed openings, alternating colors", labelpad=9)
        ax.legend(handles=[Patch(facecolor=color, label=label,
                                 hatch="///" if key == "unresolved_cutoffs" else None)
                           for key, label, color in categories],
                  frameon=False, fontsize=8.7, loc="upper left", ncol=4,
                  handlelength=1.2, columnspacing=1)
        ax.text(0, -.25, "Cutoff games remain unresolved; they are not counted as draws.\n"
                "MCTS spends more computation per move. Small match samples do not establish Elo.",
                transform=ax.transAxes, fontsize=9.2, color=MUTED, va="top", linespacing=1.5)
    else:
        unavailable(ax, "Final paired match evaluation has not completed.\nNo strength conclusion is available yet.")

    fig.text(.035, .054, "Select on validation; evaluate the selected checkpoint with and without MCTS.",
             fontsize=13, fontweight="bold")
    fig.text(.035, .026,
             f"Single seed · {config.get('hidden', '?')} channels · {str(config.get('device', '?')).upper()} · "
             "imitation and playing strength are separate objectives · no Elo estimate",
             fontsize=10, color=MUTED)
    for extension in ("png", "svg"):
        path = HERE / f"learning-comparison.{extension}"
        fig.savefig(path, dpi=180, facecolor=fig.get_facecolor())
        if extension == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
