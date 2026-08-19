"""Offline checks for the three behavioural fixes of test_223.

Runs on CPU in a couple of seconds. Verifies:
  1. the MSE step-size search now spends its candidates on the range that
     matters instead of on the +0.5 activation margin;
  2. with T2=T3=0 the METaQ perspective term is a plain ridge on the weights
     and produces a strictly one-signed push on the step sizes, which is what
     the new gate suppresses;
  3. 'relative' step-size rates give a uniform relative displacement, while
     'absolute' ones differ by the spread of the step sizes.
"""

import math
import sys

import torch

sys.path.insert(0, ".")

from utils.lsq import (  # noqa: E402
    mse_weight_step_size,
    mse_weight_step_size_per_channel,
    metaq_scale_gradient,
    signed_integer_codebook,
)

FAILURES = []


def check(name, condition, detail=""):
    status = "ok  " if condition else "FAIL"
    print(f"[{status}] {name}{(' -- ' + detail) if detail else ''}")
    if not condition:
        FAILURES.append(name)


def candidate_index(weight, q, margin):
    """Which of the 100 candidates the search lands on."""
    qp = float(q[-1].item())
    max_range = max(abs(float(weight.min())), float(weight.max())) + margin
    step = max_range / 100.0 / qp
    scale = mse_weight_step_size(weight, q, range_margin=margin)
    return scale / step


def test_mse_grid_resolution():
    q = signed_integer_codebook(16, "cpu")
    torch.manual_seed(0)
    # A weight tensor with the dynamic range of an AlexNet fully connected
    # layer: max|w| of about 0.07, which is where the +0.5 margin does damage.
    w = (torch.randn(200_000) * 0.013).clamp(-0.07, 0.07)
    old_idx = candidate_index(w, q, 0.5)
    new_idx = candidate_index(w, q, 0.0)
    check(
        "MSE search resolution improves with the default margin",
        new_idx > 4 * old_idx and new_idx > 15,
        f"candidate {old_idx:.1f}/100 with margin 0.5, {new_idx:.1f}/100 now",
    )
    check(
        "default margin of mse_weight_step_size is 0",
        mse_weight_step_size(w, q) == mse_weight_step_size(w, q, range_margin=0.0),
    )
    rows = w[:180_000].reshape(600, 300)
    check(
        "default margin of the per-channel search is 0",
        torch.equal(
            mse_weight_step_size_per_channel(rows, q),
            mse_weight_step_size_per_channel(rows, q, range_margin=0.0),
        ),
    )


def test_t1_scale_push_is_one_signed():
    """Reproduce the T1 contribution to d phi / d s at T2 = T3 = 0.

    With T2 = 0 the boundary multiplier is -T1 * v_edge and it is non-zero only
    on the clipped weights, so metaq_scale_gradient returns
    +T1 * s * sum(q_edge^2) >= 0 whatever the data. That is the bias the new
    gate removes.
    """
    t1 = 1e-5
    q = signed_integer_codebook(16, "cpu")
    qn, qp = q[0], q[-1]
    torch.manual_seed(1)
    for trial in range(200):
        scale = torch.tensor(float(torch.rand(1) * 0.1 + 1e-3))
        w = torch.randn(4096) * float(scale) * 4.0
        clipped_w = w.clamp(qn * scale, qp * scale)
        edge = torch.where(w >= 0, qp * scale, qn * scale)
        at_edge = clipped_w.abs() >= (edge.abs() - 1e-9)
        beta_constraint = torch.where(
            at_edge, -t1 * edge, torch.zeros_like(edge)
        )
        grad = metaq_scale_gradient(beta_constraint, clipped_w, scale)
        if float(grad) < 0.0:
            check("T1 step-size push is never negative", False, f"trial {trial}")
            return
    check(
        "T1 step-size push is never negative (200 random trials)",
        True,
        "always shrinks the grid, which grows the clipped fraction",
    )
    # And it really is proportional to the clipped count.
    scale = torch.tensor(0.02)
    few = torch.randn(4096) * float(scale) * 3.0
    many = torch.randn(4096) * float(scale) * 8.0
    grads = []
    fractions = []
    for w in (few, many):
        clipped_w = w.clamp(qn * scale, qp * scale)
        edge = torch.where(w >= 0, qp * scale, qn * scale)
        at_edge = clipped_w.abs() >= (edge.abs() - 1e-9)
        beta_constraint = torch.where(
            at_edge, -t1 * edge, torch.zeros_like(edge)
        )
        grads.append(float(metaq_scale_gradient(beta_constraint, clipped_w, scale)))
        fractions.append(float(at_edge.float().mean()))
    check(
        "the push grows with the clipped fraction (positive feedback loop)",
        grads[1] > 3 * grads[0] > 0,
        f"{grads[0]:.3e} at {fractions[0]:.1%} clipped, "
        f"{grads[1]:.3e} at {fractions[1]:.1%} clipped",
    )


def test_t1_fast_path_matches_general_path():
    """The T2=T3=0 fast path must reproduce the general perspective path.

    Mirrors both code paths of the trainer on the same inputs and compares the
    weight gradient they add. The general path is transcribed from
    _zero_entropy_perspective_gradients with sparsity_coeff = 0.
    """
    t1 = 1e-5
    q = signed_integer_codebook(16, "cpu")
    qn, qp = q[0], q[-1]
    torch.manual_seed(7)
    worst = 0.0
    for _ in range(50):
        n = 4096
        scale = torch.rand(n) * 0.05 + 1e-3          # per-weight step sizes
        w = torch.randn(n) * scale * 3.5

        # --- general path -----------------------------------------------------
        clipped_w = torch.minimum(torch.maximum(w, qn * scale), qp * scale)
        normalized = w / scale
        in_range = (normalized >= qn) & (normalized <= qp)
        y_star = torch.ones_like(w)                   # T2 = 0
        edge_v = torch.where(w >= 0, qp * scale, qn * scale)
        representation_floor = (
            clipped_w.abs() / edge_v.abs().clamp_min(1e-12)
        ).clamp(max=1.0)
        at_floor = (clipped_w != 0) & torch.isclose(
            y_star, representation_floor, rtol=1e-5, atol=1e-7
        )
        beta_constraint = torch.where(
            at_floor, -t1 * edge_v, torch.zeros_like(w)
        )
        beta_star = torch.where(
            clipped_w != 0,
            beta_constraint + 2.0 * t1 * clipped_w / y_star,
            torch.zeros_like(w),
        )
        general = beta_star * in_range

        # --- fast path --------------------------------------------------------
        fast = 2.0 * t1 * w * in_range

        worst = max(worst, float((general - fast).abs().max()))
    check(
        "T1-only fast path reproduces the general perspective path",
        worst < 1e-12,
        f"max absolute difference over 50 tensors: {worst:.2e}",
    )


def test_relative_scale_rate():
    """One Adam group per tensor, rate proportional to the initial step size."""
    from torch import optim

    init = [0.0042, 0.0630, 0.2150]
    base = 1e-3
    scales = [torch.nn.Parameter(torch.tensor(v)) for v in init]

    def run(groups):
        opt = optim.Adam(groups, lr=base)
        start = [float(s) for s in scales]
        for _ in range(50):
            opt.zero_grad()
            for s in scales:
                s.grad = torch.tensor(1.0)  # identical gradient everywhere
            opt.step()
        return [
            abs(float(s) - s0) / s0 for s, s0 in zip(scales, start)
        ]

    for s, v in zip(scales, init):
        s.data = torch.tensor(v)
    rel = run(
        [{"params": [s], "lr": base * v} for s, v in zip(scales, init)]
    )
    for s, v in zip(scales, init):
        s.data = torch.tensor(v)
    absol = run([{"params": scales, "lr": base}])

    check(
        "relative mode gives a uniform relative displacement",
        max(rel) / min(rel) < 1.05,
        f"spread {max(rel) / min(rel):.3f}x across step sizes 0.0042..0.215",
    )
    check(
        "absolute mode does not",
        max(absol) / min(absol) > 10,
        f"spread {max(absol) / min(absol):.1f}x on the same step sizes",
    )


if __name__ == "__main__":
    test_mse_grid_resolution()
    test_t1_scale_push_is_one_signed()
    test_t1_fast_path_matches_general_path()
    test_relative_scale_rate()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: {FAILURES}")
        raise SystemExit(1)
    print("all checks passed")
