#!/bin/sh
# Double-click in Finder, or run from any directory.
chess_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ ! -x "$chess_dir/.venv/bin/python" ]; then
    printf '%s\n' "Create the project's .venv and install requirements/gui.txt and requirements/neural.txt first."
    exit 1
fi
if [ "$#" -eq 0 ]; then
    set -- --opponent supervised --sims 64 --seconds 0
fi
exec "$chess_dir/.venv/bin/python" "$chess_dir/chess/UI_with_AI.py" "$@"
