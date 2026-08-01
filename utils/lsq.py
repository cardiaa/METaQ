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


def mse_weight_step_size(
    weight: torch.Tensor,
    q: torch.Tensor,
    num_candidates: int = 100,
    range_margin: float = 0.5,
) -> float:
    """Per-tensor symmetric MSE range search used by the reference paper.

    This mirrors its grid estimator: search 100 positive clipping thresholds
    between ``(max(abs(w)) + 0.5) / 100`` and ``max(abs(w)) + 0.5`` and choose
    the scale whose signed uniform quantization minimizes squared weight error.
    """
    if num_candidates <= 0:
        raise ValueError("num_candidates must be positive.")
    qp = float(q[-1].item())
    if qp <= 0:
        raise ValueError("The positive integer range must be non-empty.")
    with torch.no_grad():
        w = weight.detach().float()
        max_range = max(abs(float(w.min().item())), float(w.max().item()))
        max_range += float(range_margin)
        step = max_range / num_candidates
        best_error = None
        best_scale = None
        qn_tensor = q[0]
        qp_tensor = q[-1]
        for candidate in range(1, num_candidates + 1):
            scale = (step * candidate) / qp
            quantized = (w / scale).round().clamp(qn_tensor, qp_tensor) * scale
            error = (w - quantized).square().sum().item()
            if best_error is None or error < best_error:
                best_error = error
                best_scale = scale
    return max(float(best_scale), 1e-12)


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
    normalize: bool = True,
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
    gradient_scale = (
        1.0 / math.sqrt(weight.numel() * float(qp))
        if normalize
        else 1.0
    )
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
