"""Repeatable chess probes for inspecting learning, not estimating Elo.

These fixed positions are deliberately small and transparent. They are never
inserted into the training replay buffer. A search success can come from MCTS
itself, so policy preferences and raw network values are reported separately.
"""
from __future__ import annotations

from contextlib import contextmanager
import math
import random
import sys

import torch

try:
    from .chess_engine import GameState
    from .mcts import best_move, evaluate_position, mcts_search
    from .rl_interface import critic_value
except ImportError:
    from chess_engine import GameState
    from mcts import best_move, evaluate_position, mcts_search
    from rl_interface import critic_value


PROBES = (
    {
        "id": "opening",
        "label": "Initial position (no single correct move)",
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "kind": "observation",
    },
    {
        "id": "white_capture_queen",
        "label": "White can capture an undefended queen",
        "fen": "4k3/q7/8/8/8/8/8/R3K3 w - - 0 1",
        "kind": "capture",
        "expected_moves": ("a1a7",),
    },
    {
        "id": "black_capture_queen",
        "label": "Black can capture an undefended queen",
        "fen": "r3k3/8/8/8/8/8/Q7/4K3 b - - 0 1",
        "kind": "capture",
        "expected_moves": ("a8a2",),
    },
    {
        "id": "white_mate_in_one",
        "label": "White can checkmate in one move",
        "fen": "7k/8/5KQ1/8/8/8/8/8 w - - 0 1",
        "kind": "mate",
    },
    {
        "id": "black_mate_in_one",
        "label": "Black can checkmate in one move",
        "fen": "8/8/8/8/8/5kq1/8/7K b - - 0 1",
        "kind": "mate",
    },
    {
        "id": "bare_kings",
        "label": "Kings only: known draw",
        "fen": "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
        "kind": "observation",
        "expected_value": 0.0,
    },
)


def _entropy(distribution):
    return -sum(probability * math.log(probability)
                for probability in distribution.values() if probability > 0)


@contextmanager
def _preserve_evaluation_state(model):
    """Do not let optional diagnostics change the next self-play random draw."""
    python_state = random.getstate()
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    numpy = sys.modules.get("numpy")
    numpy_state = numpy.random.get_state() if numpy is not None else None
    modes = [(module, module.training) for module in model.modules()] if model is not None else []
    mps_state = None
    if model is not None and any(parameter.device.type == "mps" for parameter in model.parameters()):
        mps_state = torch.mps.get_rng_state()
    try:
        if model is not None:
            model.eval()
        yield
    finally:
        for module, mode in modes:
            module.training = mode
        random.setstate(python_state)
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        if mps_state is not None:
            torch.mps.set_rng_state(mps_state)
        if numpy_state is not None:
            numpy.random.set_state(numpy_state)


def _expected_moves(position, probe):
    legal = position.get_all_legal_moves(position.player, as_moves=True)
    if probe["kind"] == "mate":
        expected = []
        for move in legal:
            position.make_move(move)
            try:
                if position.is_checkmate():
                    expected.append(move.uci())
            finally:
                position.undo_last_move()
        if not expected:
            raise ValueError(f"Probe {probe['id']} contains no mate in one")
        return sorted(expected)
    expected = list(probe.get("expected_moves", ()))
    if probe["kind"] == "capture":
        legal_by_uci = {move.uci(): move for move in legal}
        for uci in expected:
            move = legal_by_uci.get(uci)
            if move is None or move.piece_captured[-1:] != "Q":
                raise ValueError(f"Probe {probe['id']} does not capture a queen legally")
            mover = position.player
            position.make_move(move)
            try:
                # Verify the queen was undefended: no immediate legal recapture.
                replies = position.get_all_legal_moves(position.player, as_moves=True)
                if any(reply.end == move.end and reply.piece_captured.startswith(mover) for reply in replies):
                    raise ValueError(f"Probe {probe['id']} allows a recapture")
            finally:
                position.undo_last_move()
    return expected


def evaluate_probes(model, num_sims=80):
    """Return JSON-ready raw-network and MCTS results for the fixed suite.

    Values use the side-to-move's perspective, even when that side is Black.
    Search uses no exploration noise or move sampling. `passed` is the search
    move's tactical success, not a learned-policy score or general strength.
    Terminal draw probes have no move distribution but retain the raw network
    value, making a learned draw-value error visible.
    """
    if not isinstance(num_sims, int) or num_sims < 0:
        raise ValueError("num_sims must be a nonnegative integer")
    records = []
    with _preserve_evaluation_state(model):
        for probe in PROBES:
            position = GameState.from_fen(probe["fen"])
            expected = _expected_moves(position, probe)
            value = critic_value(model, position) if model is not None else evaluate_position(position)
            if not math.isfinite(value):
                raise ValueError(f"Probe {probe['id']} has a nonfinite network value")
            search, root = mcts_search(position, model, num_sims=num_sims,
                                       add_root_dirichlet=False, return_root=True)
            priors = root._base_priors
            policy_move = best_move(priors)
            search_move = best_move(search, root)
            top_moves = sorted(search, key=lambda move: (search[move], root.edges[move].q), reverse=True)[:5]
            record = {
                "id": probe["id"],
                "label": probe["label"],
                "fen": position.to_fen(),
                "player": position.player,
                "value": float(value),
                "expected_moves": expected,
                "policy_best_move": policy_move.uci() if policy_move is not None else None,
                "search_best_move": search_move.uci() if search_move is not None else None,
                "passed": search_move is not None and search_move.uci() in expected if expected else None,
                "policy_probability_expected": sum(probability for move, probability in priors.items() if move.uci() in expected),
                "search_probability_expected": sum(probability for move, probability in search.items() if move.uci() in expected),
                "policy_entropy": _entropy(priors),
                "search_entropy": _entropy(search),
                "top_moves": [{"move": move.uci(), "prior": priors[move], "search_probability": search[move]} for move in top_moves],
            }
            if "expected_value" in probe:
                record["expected_value"] = probe["expected_value"]
            records.append(record)
    return records
