"""Learned Step Size Quantization helpers for weight-only QAT."""

from __future__ import annotations

import math

import torch


def signed_integer_codebook(num_levels: int, device) -> torch.Tensor:
    """Return the full signed integer codebook, including zero."""
    if num_levels < 2 or num_levels & (num_levels - 1):
        raise ValueError("LSQ requires a power-of-two number of levels.")
    bits = int(math.log2(num_levels))
    qn = -(2 ** (bits - 1))
    qp = 2 ** (bits - 1) - 1
    return torch.arange(qn, qp + 1, dtype=torch.float32, device=device)


def initial_weight_step_size(weight: torch.Tensor, q_positive_max: float) -> float:
    """LSQ weight-scale initialization: 2 * mean(|w|) / sqrt(Qp)."""
    if q_positive_max <= 0:
        raise ValueError("The positive LSQ integer range must be non-empty.")
    value = 2.0 * weight.detach().float().abs().mean().item()
    return max(value / math.sqrt(q_positive_max), 1e-12)


def quantize_weight(weight: torch.Tensor, scale: torch.Tensor, q: torch.Tensor):
    """Return the LSQ fake-quantized weight and its integer assignment."""
    scale_safe = scale.clamp_min(1e-12)
    normalized = weight / scale_safe
    qn = q[0]
    qp = q[-1]
    assignment = normalized.round().clamp(qn, qp)
    return assignment * scale_safe, assignment


def task_scale_gradient(
    weight: torch.Tensor,
    quantized_weight_gradient: torch.Tensor,
    scale: torch.Tensor,
    q: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the LSQ surrogate scale gradient and weight STE mask.

    The caller temporarily used the quantized values as the module parameter,
    so ``quantized_weight_gradient`` is dL/d(w_hat). This function returns the
    normalized LSQ dL/ds and the in-range mask implementing d(w_hat)/dw.
    """
    scale_safe = scale.detach().clamp_min(1e-12)
    normalized = weight.detach() / scale_safe
    qn = q[0]
    qp = q[-1]
    in_range = (normalized >= qn) & (normalized <= qp)
    d_quantized_d_scale = torch.where(
        normalized < qn,
        qn,
        torch.where(
            normalized > qp,
            qp,
            normalized.round() - normalized,
        ),
    )
    gradient_scale = 1.0 / math.sqrt(weight.numel() * float(qp))
    scale_gradient = (
        quantized_weight_gradient.detach().float()
        * d_quantized_d_scale.float()
    ).sum() * gradient_scale
    return scale_gradient, in_range


def metaq_scale_gradient(
    constraint_multiplier: torch.Tensor,
    represented_weight: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Envelope derivative d phi / d s for v_b=s*q_b.

    ``constraint_multiplier`` is the multiplier of
    w_i - sum_b s*q_b*x_i,b = 0. It is not the full dphi/dw, which also contains
    the explicit perspective-ridge derivative.
    """
    scale_safe = scale.detach().clamp_min(1e-12)
    return -(
        constraint_multiplier.detach().float()
        * represented_weight.detach().float()
    ).sum() / scale_safe
