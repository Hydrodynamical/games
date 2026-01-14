from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import your engine types
# from chess_engine import GameState, Move

# --- Action encoding: (start_square, end_square) -> int in [0, 4095] ---

def sq_to_idx(sq: Tuple[int, int]) -> int:
    r, c = sq
    return 8 * r + c

def idx_to_sq(i: int) -> Tuple[int, int]:
    return (i // 8, i % 8)

def move_to_action(mv) -> int:
    """Encode a Move as a single integer action (start->end)."""
    return sq_to_idx(mv.start) * 64 + sq_to_idx(mv.end)

def action_to_start_end(a: int) -> Tuple[Tuple[int,int], Tuple[int,int]]:
    s = a // 64
    e = a % 64
    return idx_to_sq(s), idx_to_sq(e)

NUM_ACTIONS = 64 * 64  # 4096


# --- State encoding: GameState -> tensor [C, 8, 8] ---

PIECE_TO_PLANE = {
    "wP": 0, "wN": 1, "wB": 2, "wR": 3, "wQ": 4, "wK": 5,
    "bP": 6, "bN": 7, "bB": 8, "bR": 9, "bQ": 10, "bK": 11,
}

def encode_gamestate(gs) -> torch.Tensor:
    """
    Basic encoder: 12 one-hot piece planes + 1 side-to-move plane.

    Output: float32 tensor of shape [13, 8, 8].
    """
    x = np.zeros((13, 8, 8), dtype=np.float32)

    # pieces
    for r in range(8):
        for c in range(8):
            p = gs.board[r][c]
            if p != "--":
                x[PIECE_TO_PLANE[p], r, c] = 1.0

    # side to move: fill with 1.0 if white to move else 0.0 (broadcast plane)
    if gs.player == "w":
        x[12, :, :] = 1.0

    return torch.from_numpy(x)


# --- Masking: restrict policy logits to legal moves only ---

def legal_action_mask(gs) -> torch.Tensor:
    """
    Returns a boolean mask of shape [NUM_ACTIONS] where True means "legal now".
    """
    legal_moves = gs.get_all_legal_moves(gs.player, as_moves=True)
    mask = torch.zeros(NUM_ACTIONS, dtype=torch.bool)
    for mv in legal_moves:
        mask[move_to_action(mv)] = True
    return mask


# --- Network: small CNN trunk + (policy head, value head) ---

class PolicyValueNet(nn.Module):
    def __init__(self, channels_in: int = 13, hidden: int = 128):
        super().__init__()

        # Simple CNN trunk
        self.trunk = nn.Sequential(
            nn.Conv2d(channels_in, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Policy head -> 4096 logits
        self.policy_head = nn.Sequential(
            nn.Conv2d(hidden, 32, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, NUM_ACTIONS),
        )

        # Value head -> scalar in [-1,1]
        self.value_head = nn.Sequential(
            nn.Conv2d(hidden, 32, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, 13, 8, 8]
        returns:
          policy_logits: [B, 4096]
          value: [B, 1] in [-1,1]
        """
        h = self.trunk(x)
        policy_logits = self.policy_head(h)
        value = self.value_head(h)
        return policy_logits, value

# https://chatgpt.com/share/695c8431-7344-8006-af35-f915a3159408
# --- User-facing helpers: GameState -> {Move: prob}, and GameState -> value ---

@torch.no_grad()
def policy_distribution(
    model: PolicyValueNet,
    gs,
    device: Optional[torch.device] = None,
    legal_moves: Optional[List[object]] = None,
) -> Dict[object, float]:
    """
    Returns a dict mapping each legal Move to its probability under the current policy.
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    # Encode state and run model ON DEVICE
    x = encode_gamestate(gs).unsqueeze(0).to(device)  # [1,13,8,8]
    logits, _v = model(x)  # logits: [1,4096] on device

    # Determine legal moves
    if legal_moves is None:
        legal_moves = gs.get_all_legal_moves(gs.player, as_moves=True)

    if len(legal_moves) == 0:
        return {}

    # Map legal moves to action indices (small list)
    legal_actions = [move_to_action(mv) for mv in legal_moves]
    legal_actions_t = torch.tensor(legal_actions, device=device, dtype=torch.long)

    # Gather only legal logits and softmax ON DEVICE
    legal_logits = logits[0, legal_actions_t]  # [num_legal]
    legal_probs = F.softmax(legal_logits, dim=0)

    # Return small CPU-side dict (minimal sync)
    legal_probs_cpu = legal_probs.detach().cpu().numpy()
    return {mv: float(p) for mv, p in zip(legal_moves, legal_probs_cpu)}

@torch.no_grad()
def critic_value(model: PolicyValueNet, gs, device: Optional[torch.device] = None) -> float:
    """
    Returns v(gs) in [-1,1], where +1 means current player is expected to win.
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    x = encode_gamestate(gs).unsqueeze(0).to(device)
    _logits, v = model(x)
    # Keep everything on device until the final scalar extraction.
    # (Still one sync at the end, but avoids any intermediate CPU moves.)
    return v.squeeze().detach().item()
#######################