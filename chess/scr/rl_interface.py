"""Policy/value network, rule-aware inputs, and a promotion-safe action space."""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_ACTIONS = 64 * 64
UNDERPROMOTIONS = "NBR"
NUM_ACTIONS = BASE_ACTIONS + 2 * 8 * 8 * len(UNDERPROMOTIONS)
NUM_INPUT_PLANES = 20
CHECKPOINT_VERSION = 2


def sq_to_idx(sq: Tuple[int, int]) -> int:
    return 8 * sq[0] + sq[1]


def idx_to_sq(i: int) -> Tuple[int, int]:
    return divmod(i, 8)


def move_to_action(mv) -> int:
    """Keep legacy ordinary/queen indices; append distinct N/B/R promotions."""
    promotion = (getattr(mv, "promotion", "") or "").upper()[-1:]
    if promotion in tuple(UNDERPROMOTIONS):
        side = 0 if mv.piece_moved[0] == "w" else 1
        return BASE_ACTIONS + ((side * 8 + mv.start[1]) * 8 + mv.end[1]) * 3 + UNDERPROMOTIONS.index(promotion)
    return sq_to_idx(mv.start) * 64 + sq_to_idx(mv.end)


def action_to_start_end(a: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    if not 0 <= a < NUM_ACTIONS:
        raise ValueError(f"Action must be in [0, {NUM_ACTIONS}).")
    if a < BASE_ACTIONS:
        return idx_to_sq(a // 64), idx_to_sq(a % 64)
    pair = (a - BASE_ACTIONS) // 3
    side, start_col = divmod(pair // 8, 8)
    return ((1 if side == 0 else 6), start_col), ((0 if side == 0 else 7), pair % 8)


def action_to_promotion(a: int) -> str:
    """N/B/R for extension actions; ordinary and queen moves share base actions."""
    if not 0 <= a < NUM_ACTIONS:
        raise ValueError(f"Action must be in [0, {NUM_ACTIONS}).")
    return UNDERPROMOTIONS[(a - BASE_ACTIONS) % 3] if a >= BASE_ACTIONS else ""


PIECE_TO_PLANE = {
    "wP": 0, "wN": 1, "wB": 2, "wR": 3, "wQ": 4, "wK": 5,
    "bP": 6, "bN": 7, "bB": 8, "bR": 9, "bQ": 10, "bK": 11,
}


def encode_gamestate(gs) -> torch.Tensor:
    """20 planes: pieces, side, four castling rights, EP, clock, repetition.

    Clock/repetition are clipped and scaled to the project's automatic draw
    thresholds (100 plies and three occurrences). Historical positions are not
    encoded; the rules engine still retains the full repetition history.
    """
    x = torch.zeros((NUM_INPUT_PLANES, 8, 8), dtype=torch.float32)
    for r, row in enumerate(gs.board):
        for c, piece in enumerate(row):
            if piece != "--":
                x[PIECE_TO_PLANE[piece], r, c] = 1.0
    x[12].fill_(float(gs.player == "w"))
    for plane, right in enumerate(("wK", "wQ", "bK", "bQ"), 13):
        x[plane].fill_(float(gs.castling_rights.get(right, False)))
    if gs.en_passant_target is not None:
        x[17, gs.en_passant_target[0], gs.en_passant_target[1]] = 1.0
    x[18].fill_(min(100, max(0, gs.halfmove_clock)) / 100.0)
    count = gs.position_counts.get(gs.position_key(), 1)
    x[19].fill_(min(3, max(0, count)) / 3.0)
    return x


def legal_action_mask(gs) -> torch.Tensor:
    mask = torch.zeros(NUM_ACTIONS, dtype=torch.bool)
    for mv in gs.get_all_legal_moves(gs.player, as_moves=True):
        mask[move_to_action(mv)] = True
    return mask


class PolicyValueNet(nn.Module):
    def __init__(self, channels_in: int = NUM_INPUT_PLANES, hidden: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(channels_in, hidden, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(hidden, 32, kernel_size=1), nn.ReLU(), nn.Flatten(),
            nn.Linear(32 * 8 * 8, NUM_ACTIONS),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(hidden, 32, kernel_size=1), nn.ReLU(), nn.Flatten(),
            nn.Linear(32 * 8 * 8, 256), nn.ReLU(), nn.Linear(256, 1), nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return [batch, 4480] policy logits and [batch, 1] mover values."""
        h = self.trunk(x)
        return self.policy_head(h), self.value_head(h)


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(path, device=None) -> PolicyValueNet:
    """Load a version-2 checkpoint or matching state dict, with a clear schema error."""
    device = select_device() if device is None else torch.device(device)
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    state = checkpoint.get("state_dict", checkpoint)
    try:
        trunk = state["trunk.0.weight"]
        head = state["policy_head.3.weight"]
        if trunk.shape[1] != NUM_INPUT_PLANES or head.shape[0] != NUM_ACTIONS:
            raise ValueError(
                "Checkpoint uses the old input/action encoding. Retrain with the "
                "20-plane, 4480-action model; legacy 13-plane/4096-action weights "
                "cannot represent castling state and distinct underpromotions."
            )
        model = PolicyValueNet(hidden=trunk.shape[0]).to(device)
        model.load_state_dict(state)
    except KeyError as exc:
        raise ValueError("Checkpoint does not contain a PolicyValueNet state dict.") from exc
    model.eval()
    return model


def save_model(model, path) -> None:
    torch.save({"format_version": CHECKPOINT_VERSION, "state_dict": model.state_dict()}, path)


@torch.no_grad()
def policy_distribution(model, gs, device=None, legal_moves: Optional[List[object]] = None) -> Dict[object, float]:
    legal_moves = gs.get_all_legal_moves(gs.player, as_moves=True) if legal_moves is None else legal_moves
    if not legal_moves:
        return {}
    model.eval()
    device = next(model.parameters()).device if device is None else device
    logits, _ = model(encode_gamestate(gs).unsqueeze(0).to(device))
    actions = torch.tensor([move_to_action(mv) for mv in legal_moves], device=device, dtype=torch.long)
    probabilities = F.softmax(logits[0, actions], dim=0).cpu().tolist()
    return dict(zip(legal_moves, probabilities))


@torch.no_grad()
def critic_value(model, gs, device=None) -> float:
    model.eval()
    device = next(model.parameters()).device if device is None else device
    _, value = model(encode_gamestate(gs).unsqueeze(0).to(device))
    return float(value.squeeze().item())


def temperature_distribution(dist: Dict[object, float], temperature: float) -> Dict[object, float]:
    """Apply p**(1/T) in log space; preserve zero support without underflow."""
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("Temperature must be finite and nonnegative.")
    if not dist:
        return {}
    moves, weights = list(dist), list(dist.values())
    if any(not math.isfinite(p) or p < 0 for p in weights):
        raise ValueError("Move probabilities must be finite and nonnegative.")
    if temperature == 0:
        best = max(range(len(weights)), key=weights.__getitem__)
        return {mv: float(i == best) for i, mv in enumerate(moves)}
    if not any(weights):
        return {mv: 1.0 / len(moves) for mv in moves}
    max_log = math.log(max(weights))
    scaled = [math.exp((math.log(p) - max_log) / temperature) if p else 0.0 for p in weights]
    total = sum(scaled)
    return {mv: p / total for mv, p in zip(moves, scaled)}


def pick_move_from_dist(dist: Dict[object, float], temperature: float = 0.0):
    probabilities = temperature_distribution(dist, temperature)
    if not probabilities:
        return None
    moves = list(probabilities)
    if temperature == 0:
        return max(probabilities, key=probabilities.get)
    index = torch.multinomial(torch.tensor(list(probabilities.values()), dtype=torch.float64), 1).item()
    return moves[index]
