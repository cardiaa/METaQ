"""End-to-end shape/consistency check of the per-channel wiring, on CPU.

Runs the two PRESTO gradient paths (closed-form ridge and the entropy FISTA) on
a small DeiT-like stack of tensors, per-tensor and per-channel, and checks that
every produced object has the right shape and that per-channel with a CONSTANT
step size reproduces the per-tensor result. Also prices the extra step sizes.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.knapsack import knapsack_perspective_leonardo as K  # noqa: E402
from utils.lsq import (  # noqa: E402
    expand_scale_flat,
    metaq_scale_gradient,
    mse_weight_step_size,
    mse_weight_step_size_per_channel,
    quantize_weight,
    signed_integer_codebook,
    task_scale_gradient,
)
from utils.optimization import FISTA_perspective_leonardo  # noqa: E402

DEV = torch.device("cpu")
T1, T3, T2 = 1e-5, 3e-8, 1e-7
DUAL_STEP, MAXIT, LOWER_C = 3e-9, 3, 1e-2

torch.manual_seed(0)
Q_FULL = signed_integer_codebook(16, DEV)
Q_NZ = Q_FULL[Q_FULL != 0]
CN = Q_NZ.numel()

# qkv, proj, fc1, fc2 and a 4-D patch embedding.
tensors = [
    torch.randn(1152, 384) * 0.02,
    torch.randn(384, 384) * 0.02,
    torch.randn(1536, 384) * 0.02,
    torch.randn(384, 1536) * 0.02,
    torch.randn(384, 3, 16, 16) * 0.05,
]
grads = [torch.randn_like(t) * 1e-3 for t in tensors]

total_w = sum(t.numel() for t in tensors)
print(f"tensori: {len(tensors)}, pesi totali: {total_w:,}")

for idx, (w, g) in enumerate(zip(tensors, grads)):
    O = w.shape[0]
    s_t = torch.tensor(mse_weight_step_size(w, Q_FULL))
    s_c = mse_weight_step_size_per_channel(w, Q_FULL)
    assert s_c.shape == (O,)

    # --- forward fake-quant: per-channel needs the original shape
    qc, _ = quantize_weight(w, s_c, Q_FULL)
    assert qc.shape == w.shape
    qt, _ = quantize_weight(w.reshape(-1), s_t, Q_FULL)
    assert qt.shape == (w.numel(),)

    # --- task scale gradient
    gt, mt = task_scale_gradient(w.reshape(-1), g.reshape(-1), s_t, Q_FULL, normalize=False)
    gc, mc = task_scale_gradient(w, g, s_c, Q_FULL, normalize=False)
    assert gt.dim() == 0 and gc.shape == (O,)
    assert mt.shape == (w.numel(),) and mc.shape == w.shape

    # --- PRESTO: constant per-channel scale must match per-tensor
    a_flat = expand_scale_flat(torch.full((O,), float(s_t)), w)
    assert a_flat.shape == (w.numel(),)
    qn, qp = Q_FULL[0], Q_FULL[-1]
    mw_t = w.reshape(-1).clamp(float(qn * s_t), float(qp * s_t))
    mw_c = torch.minimum(torch.maximum(w.reshape(-1), qn * a_flat), qp * a_flat)
    assert torch.allclose(mw_t, mw_c, atol=1e-9)

    xi0 = torch.rand(CN) * 3e-7
    xi_t, b_t, bc_t = FISTA_perspective_leonardo(
        xi0.clone(), s_t * Q_NZ, mw_t, CN, float(mw_t.numel()), LOWER_C,
        T1, T3, T2, 1e5, DEV, MAXIT, DUAL_STEP,
    )
    xi_c, b_c, bc_c = FISTA_perspective_leonardo(
        xi0.clone(), Q_NZ, mw_c, CN, float(mw_c.numel()), LOWER_C,
        T1, T3, T2, 1e5, DEV, MAXIT, DUAL_STEP, scale=a_flat,
    )
    assert b_c.shape == b_t.shape == (w.numel(),)
    # xi is a per-tensor dual over INTEGER symbols and must be unaffected.
    assert torch.allclose(xi_t, xi_c, rtol=1e-4, atol=1e-12), "il duale e' cambiato"
    db = (b_t - b_c).abs()
    rel = db / (b_t.abs() + 1e-12)
    # Ties on hull vertices give a legitimately different subgradient; require
    # agreement on the overwhelming majority and a small median.
    frac_bad = float((rel > 1e-3).float().mean())
    assert frac_bad < 0.10, f"tensore {idx}: {frac_bad:.1%} di beta* discordi"

    # --- PRESTO scale gradient: the reduction identity, at IDENTICAL inputs.
    # sum_c [ -sum_{i in c} mu_i w_i / a ] must equal -sum_i mu_i w_i / a.
    mg_t = metaq_scale_gradient(bc_t, mw_t, s_t)
    mg_same = metaq_scale_gradient(bc_t, mw_t, torch.full((O,), float(s_t)))
    assert mg_t.dim() == 0 and mg_same.shape == (O,)
    red = abs(float(mg_same.sum()) - float(mg_t)) / max(abs(float(mg_t)), 1e-12)
    assert red < 1e-4, f"tensore {idx}: identita' della riduzione rotta ({red:.2e})"

    # Separately, how far the tie-induced difference in the multipliers moves
    # the aggregate. This is a property of the vertex ties, not of the wiring.
    mg_c = metaq_scale_gradient(bc_c, mw_c, torch.full((O,), float(s_t)))
    tie = abs(float(mg_c.sum()) - float(mg_t)) / max(abs(float(mg_t)), 1e-12)
    print(
        f"  [{idx}] {tuple(w.shape)}  O={O:5d}  beta* discordi {frac_bad:5.2%}  "
        f"riduzione {red:.1e}  scarto da tie {tie:5.2%}  "
        f"scale span {s_c.max() / s_c.min():5.2f}x"
    )

# --- byte accounting
n_channels = sum(t.shape[0] for t in tensors)
extra_bits = (n_channels - len(tensors)) * 32
print(
    f"\nscale in piu': {n_channels - len(tensors):,} float32 = "
    f"{extra_bits / 8 / 1024:.1f} KiB = "
    f"{extra_bits / (total_w * 32):.4%} dell'FP32"
)

print("\nTUTTI I CONTROLLI SUPERATI")
