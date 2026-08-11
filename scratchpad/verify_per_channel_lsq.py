"""Offline checks for the per-channel LSQ helpers.

1. The per-tensor path must be bit-identical to the pre-change behaviour.
2. The per-channel MSE init must never be worse than the per-tensor one.
3. task_scale_gradient / metaq_scale_gradient per-channel must agree, channel by
   channel, with the per-tensor routine applied to that channel alone.
"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.lsq import (  # noqa: E402
    channel_view,
    expand_scale_flat,
    metaq_scale_gradient,
    mse_weight_step_size,
    mse_weight_step_size_per_channel,
    quantize_weight,
    signed_integer_codebook,
    task_scale_gradient,
)

torch.manual_seed(0)
DEV = torch.device("cpu")
Q = signed_integer_codebook(16, DEV)
QP = float(Q[-1].item())

# A DeiT-like tensor (384 output channels) with deliberately heterogeneous
# per-channel scales, which is the regime per-channel quantization targets.
O, K = 384, 384
base = torch.randn(O, K) * 0.02
row_gain = torch.exp(torch.linspace(-1.5, 1.5, O)).unsqueeze(1)
W = base * row_gain
W[0] += 0.6 * torch.randn(K)  # one outlier channel

# ---------------------------------------------------------------- 1. per-tensor
s_tensor = mse_weight_step_size(W, Q)
assert isinstance(s_tensor, float)
s_t = torch.tensor(s_tensor)
wq_t, _ = quantize_weight(W, s_t, Q)
mse_tensor = (W - wq_t).square().mean().item()

grad = torch.randn_like(W)
g_t, mask_t = task_scale_gradient(W.reshape(-1), grad.reshape(-1), s_t, Q, normalize=True)
assert g_t.dim() == 0, "per-tensor scale gradient must stay scalar"
assert mask_t.shape == (O * K,)

# Reference recomputation of the untouched formula.
norm = W.reshape(-1) / s_t.clamp_min(1e-12)
d = torch.where(norm < Q[0], Q[0], torch.where(norm > Q[-1], Q[-1], norm.round() - norm))
g_ref = (grad.reshape(-1) * d).sum() * (1.0 / math.sqrt(W.numel() * QP))
assert torch.equal(g_t, g_ref), "per-tensor path changed!"
print(f"[1] per-tensore invariato   scale={s_tensor:.6e}  mse={mse_tensor:.6e}")

# ------------------------------------------------------------- 2. per-channel
s_chan = mse_weight_step_size_per_channel(W, Q)
assert s_chan.shape == (O,)
wq_c, _ = quantize_weight(W, s_chan, Q)
mse_chan = (W - wq_c).square().mean().item()
assert mse_chan < mse_tensor, "per-channel must not be worse"
print(
    f"[2] per-canale  mse={mse_chan:.6e}  "
    f"riduzione {100 * (1 - mse_chan / mse_tensor):.2f}%  "
    f"scale in [{s_chan.min():.4e}, {s_chan.max():.4e}] "
    f"(rapporto {s_chan.max() / s_chan.min():.1f}x)"
)

# Each channel's step size must equal the per-tensor search run on that channel.
# Tolerance is float32 epsilon: the per-tensor helper accumulates in Python
# float64 while the vectorized per-channel search stays in float32.
worst = 0.0
for c in range(O):
    alone = mse_weight_step_size(W[c : c + 1], Q)
    worst = max(worst, abs(alone - float(s_chan[c])) / max(alone, 1e-12))
assert worst <= 1e-6, f"scarto relativo massimo {worst:.2e}"
print(
    f"[2b] init per-canale == ricerca per-tensore sul singolo canale "
    f"su tutti i {O} canali (scarto rel. max {worst:.2e})"
)

# --------------------------------------------------- 3. gradients per channel
g_c, mask_c = task_scale_gradient(W, grad, s_chan, Q, normalize=True)
assert g_c.shape == (O,)
assert mask_c.shape == W.shape
for c in (0, 3, O // 2, O - 1):
    g_one, _ = task_scale_gradient(
        W[c].reshape(-1), grad[c].reshape(-1), s_chan[c], Q, normalize=True
    )
    assert torch.allclose(g_c[c], g_one, rtol=1e-6, atol=1e-12), (
        f"task grad canale {c}: {g_c[c]} vs {g_one}"
    )
print("[3] task_scale_gradient per-canale coerente canale per canale")

mu = torch.randn(O * K) * 1e-3
wbar = W.reshape(-1)
m_c = metaq_scale_gradient(mu, wbar, s_chan)
assert m_c.shape == (O,)
for c in (0, 7, O // 2, O - 1):
    m_one = metaq_scale_gradient(
        mu.reshape(O, K)[c], W[c].reshape(-1), s_chan[c]
    )
    assert torch.allclose(m_c[c], m_one, rtol=1e-6, atol=1e-12)
print("[4] metaq_scale_gradient per-canale coerente canale per canale")

# ------------------------------------------------------------ 5. flat helpers
flat = expand_scale_flat(s_chan, W)
assert flat.shape == (O * K,)
assert torch.equal(flat.reshape(O, K)[:, 0], s_chan)
assert torch.equal(flat.reshape(O, K), s_chan.unsqueeze(1).expand(O, K))
assert channel_view(s_chan, W).shape == (O, 1)
conv = torch.randn(64, 3, 16, 16)
assert channel_view(mse_weight_step_size_per_channel(conv, Q), conv).shape == (64, 1, 1, 1)
print("[5] helper flat/broadcast corretti anche su tensori conv 4-D")

print("\nTUTTI I CONTROLLI SUPERATI")
