"""Plot archived self-play outcomes, value prediction, and equal-MCTS matches.

Run with Matplotlib installed:
    MPLCONFIGDIR=/tmp/chess-training-matplotlib python3 path/to/plot.py

The default input/output directory is beside this script. Missing measurements
remain explicitly pending. This script neither chooses checkpoints nor uses
within-training losses as held-out value estimates.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


HERE = Path(__file__).resolve().parent
INK, MUTED, GRID = "#243642", "#596A76", "#E4EAED"
BASELINE, CANDIDATE, ZERO, UNKNOWN = "#24799F", "#C57928", "#8D9DA6", "#DCE3E7"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.edgecolor": GRID, "svg.fonttype": "none"})


def read(directory, name, default=None):
    path = directory / name
    return json.loads(path.read_text()) if path.exists() else default


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def style(ax, title):
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=17)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0, pad=7)
    ax.grid(axis="y", color=GRID)


def pending(ax, message):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(.5, .5, message, transform=ax.transAxes, ha="center", va="center",
            fontsize=11, color=MUTED, linespacing=1.6)


def note(ax, text):
    ax.text(0, -.26, text, transform=ax.transAxes, fontsize=9.1, color=MUTED,
            va="top", linespacing=1.5)


def checked_curve(records):
    """A learning curve must reuse one benchmark population and reference."""
    rows = sorted(records, key=lambda row: row["game"])
    if len({row["game"] for row in rows}) != len(rows):
        raise ValueError("Duplicate checkpoint game counts in value curve")
    identity = None
    for row in rows:
        metrics = row["metrics"]
        group = metrics["all"]
        current = (group["positions"], group.get("games"),
                   metrics.get("benchmark_sha256", {}).get("games"))
        if identity is not None and current != identity:
            raise ValueError("Value-curve benchmark population or content hash changed")
        identity = current
        if rows[0] != row:
            for model in ("baseline", "zero"):
                first, value = rows[0]["metrics"]["all"][model]["mse"], group[model]["mse"]
                if (first is None) != (value is None) or (finite(first) and not math.isclose(first, value, rel_tol=1e-8, abs_tol=1e-10)):
                    raise ValueError(f"Value-curve {model} reference changed")
    return rows


def outcome_groups(values):
    if "by_game_outcome" in values:
        return values["by_game_outcome"]
    # Backward-compatible aggregation of disjoint result populations.
    results = values.get("by_game_result", {})
    output = {}
    for name, keys in (("decisive", ("1-0", "0-1")), ("draw", ("1/2-1/2",))):
        groups = [results[key] for key in keys if results.get(key, {}).get("positions", 0)]
        count = sum(group["positions"] for group in groups)
        output[name] = {"positions": count, "games": sum(group.get("games", 0) for group in groups)}
        for model in ("baseline", "candidate", "zero"):
            output[name][model] = {"mse": sum(group["positions"] * group[model]["mse"] for group in groups) / count if count else None}
    return output


def checked_matches(matches):
    if not matches.get("games"):
        return False
    if matches["candidate_wins"] + matches["baseline_wins"] + matches["actual_draws"] != matches["completed_games"]:
        raise ValueError("Completed match counts do not reconcile")
    if matches["completed_games"] + matches["unresolved_cutoffs"] != matches["games"]:
        raise ValueError("Unknown cutoffs are not represented in total match count")
    points = matches["candidate_wins"] + .5 * matches["actual_draws"]
    expected = points / matches["completed_games"] if matches["completed_games"] else None
    if matches["candidate_score_on_completed_games"] != expected:
        raise ValueError("Match score does not use the completed-game denominator")
    bounds = [points / matches["games"], (points + matches["unresolved_cutoffs"]) / matches["games"]]
    if matches.get("candidate_score_all_games_bounds") != bounds:
        raise ValueError("All-game score bounds do not account for unresolved games")
    return True


def make_figure(directory):
    directory = Path(directory)
    protocol = read(directory, "protocol.json", {})
    summary = read(directory, "summary.json", {})
    comparison = read(directory, "comparison/summary.json", {})
    values, matches = comparison.get("values", {}), comparison.get("matches", {})
    curve = checked_curve(read(directory, "value_curve.json", []))
    retention = read(directory, "human_retention.json", {})
    target_games = protocol.get("training", {}).get("iterations")
    completed = summary.get("games_completed")
    has_final = comparison.get("status") == "completed"
    if has_final and curve and target_games is not None:
        final_point = next((row for row in curve if row["game"] == target_games), None)
        expected_hash = comparison.get("config", {}).get("input_sha256", {}).get("candidate_checkpoint")
        if final_point and expected_hash and final_point.get("checkpoint_sha256") != expected_hash:
            raise ValueError("The plotted final checkpoint is not the match candidate")

    fig = plt.figure(figsize=(15, 11.2), facecolor="#FAFCFD")
    grid = fig.add_gridspec(2, 2, left=.09, right=.96, top=.79, bottom=.20,
                           hspace=.66, wspace=.30)
    title = "Does MCTS self-play improve the supervised engine?"
    if has_final and finite(matches.get("candidate_score_on_completed_games")) and matches["candidate_score_on_completed_games"] < .5:
        title = "This self-play checkpoint lost to the supervised baseline"
    fig.text(.035, .953, title, fontsize=23, fontweight="bold")
    budget = f"{target_games} games" if target_games is not None else "the predeclared game budget"
    fig.text(.035, .912, f"Supervised baseline → MCTS self-play · candidate fixed after {budget} · no test-based checkpoint selection",
             fontsize=11.4, color=MUTED)
    if summary:
        training_line = (f"Training: {summary.get('status', 'not recorded')} · {completed if completed is not None else '—'} games · "
                         f"{summary.get('decisive_games', '—')} decisive / {summary.get('actual_draws', '—')} draws / "
                         f"{summary.get('move_limit_games', '—')} move-limit cutoffs")
    else:
        training_line = "Training summary pending · measurements will appear only when recorded"
    fig.text(.035, .877, training_line, fontsize=11, color=MUTED)
    if has_final and finite(matches.get("candidate_score_on_completed_games")):
        finding = f"Candidate match score: {matches['candidate_score_on_completed_games']:.1%}. "
        if values.get("all", {}).get("candidate", {}).get("mse", math.inf) > values.get("all", {}).get("zero", {}).get("mse", math.inf):
            finding += "Its benchmark value MSE remains worse than always predicting zero."
        if matches["candidate_score_on_completed_games"] < .5:
            finding += " Retain the supervised baseline."
        fig.text(.035, .841, finding, fontsize=10.7, color="#975D22")

    ax = fig.add_subplot(grid[0, 0])
    style(ax, "A   Value prediction on the same held-out games")
    points = [row for row in curve if finite(row["metrics"]["all"]["candidate"]["mse"])]
    if points:
        x = [row["game"] for row in points]
        y = [row["metrics"]["all"]["candidate"]["mse"] for row in points]
        first = points[0]["metrics"]
        baseline, zero = first["all"]["baseline"]["mse"], first["all"]["zero"]["mse"]
        ax.plot(x, y, color=CANDIDATE, marker="o", linewidth=2.2, markersize=4.5, label="Self-play checkpoint")
        ax.axhline(baseline, color=BASELINE, linewidth=1.8, linestyle="--", label="Frozen supervised baseline")
        ax.axhline(zero, color=ZERO, linewidth=1.8, linestyle=":", label="Always-zero prediction")
        ax.set_xlim(min(x) - 3, max(max(x) + 5, target_games or 0))
        ax.set_ylim(0, max(y + [baseline, zero, .05]) * 1.55)
        ax.set_xticks(x)
        ax.set_xlabel("Self-play training games", labelpad=8)
        ax.set_ylabel("Value MSE · lower is better", labelpad=8)
        ax.annotate(f"{y[-1]:.3f}", (x[-1], y[-1]), xytext=(0, -8), textcoords="offset points", ha="center", va="top", color=CANDIDATE, fontsize=9)
        ax.legend(frameon=False, fontsize=8.6, loc="upper left")
        minimum = min(points, key=lambda row: row["metrics"]["all"]["candidate"]["mse"])
        note(ax, f"{first['all']['positions']:,} positions in {first['all']['games']} completed games; "
             f"{first.get('excluded_unknown_outcome_games', '—')} cutoffs excluded.\n"
             f"Lowest observed MSE: {minimum['metrics']['all']['candidate']['mse']:.3f} at game {minimum['game']}; not used for selection.")
    else:
        pending(ax, "Fixed-checkpoint value curve pending.\nNo training-loss values are substituted.")

    ax = fig.add_subplot(grid[0, 1])
    style(ax, "B   Value errors: decisive games and draws")
    groups = outcome_groups(values)
    available = [(key, groups[key]) for key in ("decisive", "draw") if groups.get(key, {}).get("positions", 0)]
    if available:
        x, maxima = np.arange(len(available)), []
        for model, label, color, offset in (("baseline", "Supervised", BASELINE, -.25),
                                           ("candidate", "Self-play", CANDIDATE, 0), ("zero", "Always zero", ZERO, .25)):
            numbers = [group[model]["mse"] for _, group in available]
            if not all(finite(value) for value in numbers):
                raise ValueError("Missing outcome-specific value MSE")
            maxima.extend(numbers)
            ax.bar(x + offset, numbers, width=.22, color=color, label=label)
            for position, value in zip(x + offset, numbers):
                ax.annotate(f"{value:.3f}", (position, value), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8.6)
        ax.set_xticks(x, [f"{key.title()}\n{group['positions']:,} positions" for key, group in available])
        ax.set_ylim(0, max(maxima + [.05]) * 1.3)
        ax.set_ylabel("Value MSE · lower is better", labelpad=8)
        ax.legend(frameon=False, fontsize=8.6, loc="upper right")
        ci = values.get("bootstrap_game_level", {}).get("candidate_minus_baseline_mse")
        delta = values.get("all", {}).get("candidate_minus_baseline_mse")
        change = f"Overall ΔMSE (self-play − supervised): {delta:+.3f}." if finite(delta) else ""
        interval = f"Paired game-bootstrap 95% interval: [{ci[0]:+.3f}, {ci[1]:+.3f}]." if ci else "Too few completed games for a paired interval."
        shrinkage = ""
        means = [values.get("all", {}).get(model, {}).get("mean_absolute_prediction") for model in ("baseline", "candidate")]
        if all(finite(value) for value in means):
            shrinkage = f"\nMean absolute predicted value: {means[0]:.3f} → {means[1]:.3f}."
        note(ax, change + "\n" + interval + shrinkage)
    else:
        pending(ax, "Completed-game value comparison pending.\nDecisive and draw populations stay separate.")

    ax = fig.add_subplot(grid[1, 0])
    style(ax, "C   Equal-MCTS matches between the two models")
    if has_final and checked_matches(matches):
        counts = [matches[key] for key in ("candidate_wins", "baseline_wins", "actual_draws", "unresolved_cutoffs")]
        bars = ax.bar(range(4), counts, width=.62, color=[CANDIDATE, BASELINE, ZERO, UNKNOWN])
        bars[-1].set_hatch("///")
        bars[-1].set_edgecolor("#A9B8C0")
        for bar, count in zip(bars, counts):
            ax.annotate(str(count), (bar.get_x() + bar.get_width() / 2, count), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=12)
        ax.set_xticks(range(4), ["Self-play\nwins", "Supervised\nwins", "Actual\ndraws", "Unresolved\ncutoffs"])
        ax.set_ylim(0, max(counts + [1]) * 1.35)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylabel(f"Games · {matches['games']} total, colors paired", labelpad=8)
        score, bounds = matches["candidate_score_on_completed_games"], matches["candidate_score_all_games_bounds"]
        score_text = f"Self-play score: {score:.1%} over {matches['completed_games']} completed games." if finite(score) else "No completed games; completed-game score is undefined."
        note(ax, score_text + f"\nAll-game score bounds: {bounds[0]:.1%}–{bounds[1]:.1%}; cutoffs are not draws.")
    else:
        pending(ax, "Completed equal-search match set pending.\nUnresolved games will remain separate.")

    ax = fig.add_subplot(grid[1, 1])
    style(ax, "D   Human-game imitation after self-play")
    before = retention.get("baseline", {}).get("metrics", {})
    after = retention.get("candidate", {}).get("metrics", {})
    if before and after:
        for key in ("policy_sample_count", "value_sample_count"):
            if before.get(key) != after.get(key):
                raise ValueError("Human-test populations differ between models")
        ax.axis("off")
        metrics = [("Top-one move agreement", "policy_top1_full_action", True),
                   ("Top-five move agreement", "policy_top5_full_action", True),
                   ("Move negative log likelihood ↓", "policy_nll_full_action", False),
                   ("Human-outcome value MSE ↓", "value_mse", False),
                   ("Human-outcome value MAE ↓", "value_mae", False),
                   ("Always-zero value MSE", "zero_value_baseline_mse", False)]
        cells = []
        for label, key, percent in metrics:
            numbers = [before.get(key), after.get(key)]
            cells.append([label] + [f"{value:.1%}" if percent else f"{value:.3f}" for value in numbers])
        table = ax.table(cellText=cells, colLabels=["Full human test split", "Supervised", "Self-play"],
                         colWidths=[.59, .205, .205], cellLoc="left", loc="upper left", bbox=[0, .12, 1, .86])
        table.auto_set_font_size(False)
        table.set_fontsize(9.2)
        for (row, column), cell in table.get_celld().items():
            cell.set_edgecolor(GRID)
            cell.set_linewidth(.6)
            cell.set_facecolor("#EDF3F6" if row == 0 else "white")
            if row == 0:
                cell.set_text_props(weight="bold", color=INK)
            elif column > 0:
                cell.set_text_props(color=BASELINE if column == 1 else CANDIDATE)
        note(ax, f"Policy: {before['policy_sample_count']:,} Hikaru turns across all 4,480 actions.\n"
             f"Values: {before['value_sample_count']:,} positions from both sides. Imitation is not strength.")
    else:
        pending(ax, "Full human-test retention evaluation pending.\nThis set is not used to select the candidate.")

    sims = comparison.get("config", {}).get("sims", protocol.get("strength_comparison", {}).get("sims"))
    search_note = (f"Matches use {sims} simulations per move for both models, fresh roots, deterministic choices, and no exploration noise."
                   if sims is not None else "Matches use the same search budget and deterministic settings for both models.")
    fig.text(.035, .10, "Value targets are outcomes of the frozen supervised baseline playing exploratory self-play, not perfect-play evaluations.",
             fontsize=9.8, color=MUTED)
    fig.text(.035, .070, search_note, fontsize=9.8, color=MUTED)
    fig.text(.035, .040, "Value accuracy, human imitation, and playing strength are distinct. One small paired match set does not establish an Elo rating.",
             fontsize=9.8, color=MUTED)
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, default=HERE)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    destination = args.output_dir or args.archive_dir
    destination.mkdir(parents=True, exist_ok=True)
    figure = make_figure(args.archive_dir)
    for extension in ("png", "svg"):
        output = destination / f"selfplay-comparison.{extension}"
        figure.savefig(output, dpi=170, facecolor=figure.get_facecolor())
        if extension == "svg":
            output.write_text("\n".join(line.rstrip() for line in output.read_text().splitlines()) + "\n")
        print(output)
    plt.close(figure)


if __name__ == "__main__":
    main()
