# https://chatgpt.com/share/695c85a9-4ed4-8006-83ab-d4bbc3e7b500

# train.py
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.optim import Adam

from rl_interface import PolicyValueNet, NUM_ACTIONS


def train_step(
    model,
    optimizer,
    batch,
    device,
    value_loss_weight: float = 1.0,
):
    """
    batch: list of (state_tensor, pi_vector, z)
    """
    model.train()

    states = torch.stack([s for (s, _, _) in batch]).to(device)
    pi_targets = torch.stack([pi for (_, pi, _) in batch]).to(device)
    z_targets = torch.tensor([z for (_, _, z) in batch], dtype=torch.float32).to(device)

    logits, values = model(states)
    values = values.squeeze(1)

    # Policy loss (cross entropy on full action space)
    log_probs = F.log_softmax(logits, dim=1)
    policy_loss = -(pi_targets * log_probs).sum(dim=1).mean()

    # Value loss
    value_loss = F.mse_loss(values, z_targets)

    loss = policy_loss + value_loss_weight * value_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {
        "loss": loss.item(),
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
    }
