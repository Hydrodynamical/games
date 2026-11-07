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
    *,
    value_mask=None,
):
    """Train on (state, search policy, outcome) examples and return metrics.

    Predictions, losses, and agreement describe the model *before* the update.
    Policy entropy and KL use natural logarithms; the model distribution is over
    the full action space, as in the policy training loss. Standard deviations
    are population statistics, including for a batch containing one position.
    ``value_mask`` selects examples with known outcomes. Policy training always
    uses every example; value loss and value diagnostics use only selected rows.
    Value loss is normalized by the number selected, not the full batch size.
    An empty selection produces zero value loss and unavailable (None) value
    diagnostics. Its disconnected value loss leaves value-head gradients None,
    so optimizer momentum cannot update that head without supervision.
    Gradient norm is measured after backward and before the optimizer step.
    """
    if not batch:
        raise ValueError("Training requires at least one position.")
    model.train()

    states = torch.stack([s for (s, _, _) in batch]).to(device)
    pi_targets = torch.stack([pi for (_, pi, _) in batch]).to(device)
    z_targets = torch.tensor([z for (_, _, z) in batch], dtype=torch.float32).to(device)

    if value_mask is None:
        eligible = torch.ones(len(batch), device=device, dtype=torch.bool)
    else:
        try:
            eligible = torch.as_tensor(value_mask, device=device)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError("value_mask must be a one-dimensional sequence of booleans or 0/1 values.") from exc
        if eligible.ndim != 1 or eligible.shape[0] != len(batch):
            raise ValueError("value_mask must have one entry per batch example.")
        if eligible.is_complex() or not torch.isfinite(eligible).all() or not ((eligible == 0) | (eligible == 1)).all():
            raise ValueError("value_mask entries must be booleans or finite 0/1 values.")
        eligible = eligible.to(dtype=torch.bool)
    value_count = int(eligible.sum().item())

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

    # Preserve the original reduction exactly when the entire batch is eligible.
    selected_values = values if value_count == len(batch) else values[eligible]
    selected_targets = z_targets if value_count == len(batch) else z_targets[eligible]
    value_loss = (F.mse_loss(selected_values, selected_targets) if value_count
                  else values.new_zeros(()))

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
            "policy_sample_count": len(batch),
            "value_sample_count": value_count,
            "value_sample_fraction": value_count / len(batch),
            "value_metrics_available": bool(value_count),
            "predicted_value_mean": selected_values.mean().item() if value_count else None,
            "predicted_value_std": selected_values.std(unbiased=False).item() if value_count else None,
            "target_value_mean": selected_targets.mean().item() if value_count else None,
            "target_value_std": selected_targets.std(unbiased=False).item() if value_count else None,
            "value_mae": (selected_values - selected_targets).abs().mean().item() if value_count else None,
            "zero_value_baseline_mse": selected_targets.square().mean().item() if value_count else None,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
    if not all(value is None or math.isfinite(value) for value in metrics.values()):
        raise ValueError("Training metrics must be finite before updating weights.")

    optimizer.zero_grad(set_to_none=True)
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
