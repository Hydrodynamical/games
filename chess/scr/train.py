"""One optimizer update with interpretable, pre-update learning diagnostics."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def train_step(
    model,
    optimizer,
    batch,
    device,
    value_loss_weight: float = 1.0,
):
    """Train on (state, search policy, outcome) examples and return metrics.

    Predictions, losses, and agreement describe the model *before* the update.
    Policy entropy and KL use natural logarithms; the model distribution is over
    the full action space, as in the policy training loss. Standard deviations
    are population statistics, including for a batch containing one position.
    Gradient norm is measured after backward and before the optimizer step.
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
    if not torch.isfinite(logits).all() or not torch.isfinite(values).all():
        raise ValueError("Model predictions must be finite before updating weights.")

    # Policy loss (cross entropy on full action space)
    log_probs = F.log_softmax(logits, dim=1)
    policy_loss = -(pi_targets * log_probs).sum(dim=1).mean()

    # Value loss
    value_loss = F.mse_loss(values, z_targets)

    loss = policy_loss + value_loss_weight * value_loss

    with torch.no_grad():
        target_entropy = -(pi_targets * pi_targets.clamp_min(1e-30).log()).sum(dim=1).mean()
        metrics = {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "policy_target_entropy": target_entropy.item(),
            # Roundoff can make exact agreement differ from zero by a few ulps.
            "policy_kl": max(0.0, (policy_loss - target_entropy).item()),
            "policy_top1_agreement": (logits.argmax(dim=1) == pi_targets.argmax(dim=1)).float().mean().item(),
            "predicted_value_mean": values.mean().item(),
            "predicted_value_std": values.std(unbiased=False).item(),
            "target_value_mean": z_targets.mean().item(),
            "target_value_std": z_targets.std(unbiased=False).item(),
            "value_mae": (values - z_targets).abs().mean().item(),
            "zero_value_baseline_mse": z_targets.square().mean().item(),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError("Training metrics must be finite before updating weights.")

    optimizer.zero_grad()
    loss.backward()
    # Observational only: do not clip, rescale, or otherwise change gradients.
    metrics["gradient_norm"] = math.sqrt(sum(
        float(parameter.grad.detach().float().norm().item()) ** 2
        for parameter in model.parameters() if parameter.grad is not None
    ))
    if not math.isfinite(metrics["gradient_norm"]):
        optimizer.zero_grad()
        raise ValueError("Gradients must be finite before updating weights.")
    optimizer.step()

    return metrics
