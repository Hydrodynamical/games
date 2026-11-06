"""Build a portable, offline HTML inspector from a training run's JSON logs.

Usage: python chess/scr/report.py path/to/run [--output path/to/report.html]
No plotting, web server, or neural-network dependencies are required.
"""
from __future__ import annotations

import argparse
import json
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
