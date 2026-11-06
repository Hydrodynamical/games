# https://chatgpt.com/share/695c85a9-4ed4-8006-83ab-d4bbc3e7b500

# train.py
from __future__ import annotations

import torch
import torch.nn.functional as F


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
    if not batch:
        raise ValueError("Training requires at least one position.")
    model.train()

    states = torch.stack([s for (s, _, _) in batch]).to(device)
    pi_targets = torch.stack([pi for (_, pi, _) in batch]).to(device)
    z_targets = torch.tensor([z for (_, _, z) in batch], dtype=torch.float32).to(device)

    if not torch.isfinite(pi_targets).all() or (pi_targets < 0).any():
        raise ValueError("Policy targets must be finite and nonnegative.")
    if not torch.allclose(pi_targets.sum(dim=1), torch.ones(len(batch), device=device), atol=1e-5):
        raise ValueError("Each policy target must sum to one.")
    if not torch.isfinite(z_targets).all() or (z_targets.abs() > 1).any():
        raise ValueError("Value targets must be finite and in [-1, 1].")

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
