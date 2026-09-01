"""Learned Step Size Quantization helpers for weight-only QAT."""

from __future__ import annotations

import math

import torch


def is_per_channel(scale: torch.Tensor) -> bool:
    """True when ``scale`` holds one step size per output channel."""
    return isinstance(scale, torch.Tensor) and scale.dim() > 0


def channel_view(scale: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Reshape a per-channel ``scale`` so that it broadcasts against ``weight``.

    Output channels are the leading dimension of every weight tensor handled
    here, so the step sizes are viewed as ``(O, 1, ..., 1)``. A scalar scale is
    returned unchanged and therefore broadcasts as before.
    """
    if not is_per_channel(scale):
        return scale
    if scale.numel() != weight.shape[0]:
        raise ValueError(
            f"Per-channel scale has {scale.numel()} entries but the weight has "
            f"{weight.shape[0]} output channels."
        )
    return scale.reshape(-1, *([1] * (weight.dim() - 1)))


def expand_scale_flat(scale: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Return one step size per weight, matching ``weight.reshape(-1)``.

    Several call sites flatten the tensor before handing it to the PRESTO
    solver. Since output channels are the leading dimension, the flat layout
    repeats each channel step size ``weight.numel() // O`` times.
    """
    if not is_per_channel(scale):
        return scale
    per_channel_count = weight.numel() // weight.shape[0]
    return scale.repeat_interleave(per_channel_count)


def channel_rows(flat: torch.Tensor, num_channels: int) -> torch.Tensor:
    """View a flat per-weight vector as ``(O, -1)`` for per-channel reductions."""
    return flat.reshape(num_channels, -1)


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
    range_margin: float = 0.0,
) -> float:
    """Per-tensor symmetric MSE range search.

    Search ``num_candidates`` positive clipping thresholds between
    ``(max(abs(w)) + margin) / num_candidates`` and ``max(abs(w)) + margin`` and
    choose the scale whose signed uniform quantization minimizes the squared
    weight error.

    The reference implementation uses ``range_margin=0.5`` because it searches
    ACTIVATION ranges, which are O(1). On weights that additive margin destroys
    the resolution of the search: with 4 bits the optimal step size sits at
    roughly ``0.4 * max|w| / q_p``, so the winning candidate has index
    ``~40 * max|w| / (max|w| + margin)`` out of ``num_candidates``. Measured on
    the real AlexNet weights with ``margin=0.5``: index 4/100 for
    ``classifier.1`` (37.7M parameters, 25% resolution on the step size), 5/100
    for ``classifier.4``, 8/100 for ``features.3``. Per output channel it is
    worse still, since narrow channels have an even smaller ``max|w|``. The
    default is therefore 0, which spreads the candidates over the range that
    actually matters.
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


def mse_weight_step_size_per_channel(
    weight: torch.Tensor,
    q: torch.Tensor,
    num_candidates: int = 100,
    range_margin: float = 0.0,
) -> torch.Tensor:
    """Per-output-channel version of :func:`mse_weight_step_size`.

    Runs the same grid of ``num_candidates`` clipping thresholds, but keeps one
    independent search per output channel and returns the winning step sizes as
    a tensor of shape ``(O,)``. The candidate grid is built from each channel's
    own weight range, which is the entire point of the per-channel quantizer:
    channels whose weights span a narrow interval no longer have to share a
    step size with the widest channel in the tensor. See
    :func:`mse_weight_step_size` for why ``range_margin`` defaults to 0.
    """
    if num_candidates <= 0:
        raise ValueError("num_candidates must be positive.")
    qp = float(q[-1].item())
    if qp <= 0:
        raise ValueError("The positive integer range must be non-empty.")
    with torch.no_grad():
        rows = weight.detach().float().reshape(weight.shape[0], -1)
        max_range = torch.maximum(
            rows.min(dim=1).values.abs(), rows.max(dim=1).values
        ) + float(range_margin)
        step = (max_range / num_candidates).unsqueeze(1)
        qn_tensor = q[0]
        qp_tensor = q[-1]
        best_error = None
        best_scale = None
        for candidate in range(1, num_candidates + 1):
            scale = (step * candidate) / qp
            quantized = (rows / scale).round().clamp(qn_tensor, qp_tensor) * scale
            error = (rows - quantized).square().sum(dim=1)
            if best_error is None:
                best_error = error
                best_scale = scale.squeeze(1).clone()
            else:
                improved = error < best_error
                best_error = torch.where(improved, error, best_error)
                best_scale = torch.where(
                    improved, scale.squeeze(1), best_scale
                )
    return best_scale.clamp_min(1e-12)


def quantize_weight(weight: torch.Tensor, scale: torch.Tensor, q: torch.Tensor):
    """Return the LSQ fake-quantized weight and its integer assignment.

    With a per-channel ``scale`` the weight must be passed in its original
    shape, so that output channels remain the leading dimension.
    """
    scale_safe = scale.clamp_min(1e-12)
    if is_per_channel(scale_safe):
        scale_safe = channel_view(scale_safe, weight)
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
    scale_detached = scale.detach().clamp_min(1e-12)
    per_channel = is_per_channel(scale_detached)
    scale_safe = (
        channel_view(scale_detached, weight) if per_channel else scale_detached
    )
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
    product = (
        quantized_weight_gradient.detach().float()
        * d_quantized_d_scale.float()
    )
    if per_channel:
        # One independent step size per output channel, so the LSQ gradient
        # scale uses that channel's element count rather than the tensor's.
        rows = product.reshape(weight.shape[0], -1)
        gradient_scale = (
            1.0 / math.sqrt(rows.shape[1] * float(qp)) if normalize else 1.0
        )
        scale_gradient = rows.sum(dim=1) * gradient_scale
    else:
        gradient_scale = (
            1.0 / math.sqrt(weight.numel() * float(qp)) if normalize else 1.0
        )
        scale_gradient = product.sum() * gradient_scale
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
    product = -(
        constraint_multiplier.detach().float()
        * represented_weight.detach().float()
    )
    if is_per_channel(scale_safe):
        # Both inputs arrive flattened from the solver; output channels are the
        # leading dimension, so the flat vector reshapes into one row each.
        return channel_rows(product, scale_safe.numel()).sum(dim=1) / scale_safe
    return product.sum() / scale_safe
