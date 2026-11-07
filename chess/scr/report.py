"""Build a portable, offline HTML inspector from a training run's JSON logs.

Usage: python chess/scr/report.py path/to/run [--output path/to/report.html]
No plotting, web server, or neural-network dependencies are required.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import tempfile


def _reject_nonfinite(value):
    raise ValueError(f"Non-finite JSON number: {value}")


def _read_json(path: Path, warnings: list[str]) -> dict:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_nonfinite)
        if not isinstance(value, dict):
            raise ValueError("expected an object")
        return value
    except (OSError, ValueError) as exc:
        warnings.append(f"Could not read {path.name}: {exc}")
        return {}


def _read_jsonl(path: Path, warnings: list[str]) -> list[dict]:
    if not path.exists():
        return []
    records = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        warnings.append(f"Could not read {path.name}: {exc}")
        return records
    for index, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line, parse_constant=_reject_nonfinite)
            if not isinstance(record, dict):
                raise ValueError("expected an object")
            records.append(record)
        except ValueError:
            warnings.append(f"Skipped incomplete or invalid {path.name} line {index}.")
    return records


def _finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _value_display_rows(updates, cutoff_policy="draw"):
    """Separate loss numerators from the examples actually supervising value.

    Preserve raw records elsewhere. A policy-only batch has a zero contribution
    to total loss, but it has no measured value MSE or target statistics.
    """
    rows = []
    value_metrics = ("value_loss", "value_mae", "predicted_value_mean", "predicted_value_std",
                     "target_value_mean", "target_value_std", "zero_value_baseline_mse")
    for update in updates:
        row = dict(update)
        policy_count = row.get("policy_sample_count", row.get("batch_size"))
        count = row.get("value_sample_count")
        fraction = row.get("value_sample_fraction")
        if not _finite_number(fraction):
            if _finite_number(count) and _finite_number(policy_count) and policy_count > 0:
                fraction = count / policy_count
            elif cutoff_policy == "exclude" and _finite_number(row.get("cutoff_target_fraction")):
                fraction = 1.0 - row["cutoff_target_fraction"]
            elif cutoff_policy == "draw":
                fraction = 1.0
            else:
                fraction = None
        if not _finite_number(count) and _finite_number(policy_count) and _finite_number(fraction):
            count = policy_count * fraction
        available = row.get("value_metrics_available") is not False
        if (_finite_number(count) and count == 0) or (_finite_number(fraction) and fraction == 0):
            available = False
        row.update(policy_sample_count=policy_count, value_sample_count=count,
                   value_sample_fraction=fraction, value_metrics_available=available,
                   policy_only_fraction=1.0 - fraction if _finite_number(fraction) else None)
        if not available:
            row.update({key: None for key in value_metrics})
        elif not _finite_number(row.get("zero_value_baseline_mse")):
            win, loss = row.get("target_win_fraction"), row.get("target_loss_fraction")
            baseline = None
            if _finite_number(win) and _finite_number(loss):
                if cutoff_policy == "exclude" and _finite_number(fraction) and fraction > 0:
                    baseline = (win + loss) / fraction
                elif cutoff_policy == "draw" and fraction == 1:
                    baseline = win + loss
            row["zero_value_baseline_mse"] = baseline
        rows.append(row)
    return rows


def generate_report(run_dir, output=None) -> Path:
    """Create a self-contained report, tolerating missing or in-progress logs.

    Invalid records are skipped with a visible warning. Log strings are embedded
    as escaped JSON data and displayed through textContent, never as HTML.
    """
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise ValueError(f"Training run directory does not exist: {run_dir}")
    warnings: list[str] = []
    data = {
        "run_name": run_dir.name,
        "metadata": _read_json(run_dir / "config.json", warnings),
        "games": _read_jsonl(run_dir / "games.jsonl", warnings),
        "updates": _read_jsonl(run_dir / "updates.jsonl", warnings),
        "probes": _read_jsonl(run_dir / "probes.jsonl", warnings),
        "summary": _read_json(run_dir / "summary.json", warnings),
        "warnings": warnings,
    }
    cutoff_policy = data["metadata"].get("config", {}).get("cutoff_value_policy", "draw")
    data["value_updates"] = _value_display_rows(data["updates"], cutoff_policy)
    # A literal </script> closes even an application/json script element.
    payload = json.dumps(data, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    payload = payload.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    template = Path(__file__).with_name("report_template.html").read_text(encoding="utf-8")
    document = template.replace("__TRAINING_DATA__", payload)
    destination = Path(output).resolve() if output else run_dir / "report.html"
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Distinct temporary names also allow a manual refresh while the runner
    # generates its own snapshot, without racing over the same temporary file.
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=destination.parent,
                                     prefix=destination.name + ".", suffix=".tmp",
                                     delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(document)
    try:
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        print(generate_report(args.run_dir, args.output))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
