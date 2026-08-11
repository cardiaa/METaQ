"""Verify the per-channel generalization of knapsack_perspective_leonardo.

A. REGRESSION  scale=None must be bit-identical to the previous behaviour.
B. EQUIVALENCE per-channel with a CONSTANT scale a must reproduce the
   per-tensor call made with the real levels v = a*q. Isolates the rescaling.
C. OPTIMALITY  per-channel with scales spanning two orders of magnitude must
   match cvxpy on the ORIGINAL problem, per weight, with its own real levels
   v_b = a_i q_b. No normalization anywhere in the reference.
D. GRADIENT    beta_star must equal dPhi/dw, checked by central differences on
   the cvxpy optimal VALUE. The duals returned by cvxpy are NOT usable here:
   the problem is badly scaled (xi ~ 3e-7, T1 = 1e-5) and ECOS and CLARABEL
   disagree with each other on them, while both agree on the value to 1e-8.
"""
import os
import sys
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.knapsack import knapsack_perspective_leonardo as K  # noqa: E402

try:
    import cvxpy as cp
except ImportError:
    cp = None

DEV = torch.device("cpu")
T1, T3, T2 = 1e-5, 3e-8, 1e-7          # perspective, entropy, sparsity
C16 = torch.arange(-8, 8, dtype=torch.float32)
Q = C16[C16 != 0]                       # 15 non-zero integer levels
CN = Q.numel()


def symbol_masses(x_placeholder, n_sym):
    """Dense mass per symbol: the representation-invariant primal object.

    At a hull VERTEX the optimum can be written on either incident facet, so
    the index pair differs while the mass per symbol does not.
    """
    m = torch.zeros(x_placeholder.shape[0], n_sym)
    rows = torch.arange(x_placeholder.shape[0])
    m[rows, x_placeholder[:, 0].long()] += x_placeholder[:, 2]
    m[rows, x_placeholder[:, 1].long()] += x_placeholder[:, 3]
    return m


# ------------------------------------------------------------ A. regression
g = torch.Generator().manual_seed(1)
xi = torch.rand(CN, generator=g) * 3e-7
a_const = 0.0184
torch.manual_seed(7)
w = torch.randn(4000) * 0.03

ref = K(xi, a_const * Q, w, CN, DEV, T1, T3, T2)
again = K(xi, a_const * Q, w, CN, DEV, T1, T3, T2, scale=None)
for r, s, name in zip(ref, again, ("x", "beta_star", "y_star", "beta_old")):
    assert torch.equal(r, s), f"regressione su {name}"
print("[A] scale=None bit-identico al comportamento precedente")

# ----------------------------------------------------------- B. equivalence
pc = K(xi, Q, w, CN, DEV, T1, T3, T2, scale=torch.full_like(w, a_const))

y_gap = (ref[2] - pc[2]).abs().max()
assert y_gap < 1e-6, f"y*: max |d| = {y_gap:.3e}"
# float32 resolution on masses of order 1. The per-tensor path lands on
# 6.0000005 instead of 6 and leaks ~3e-7 onto the neighbouring symbol; the
# per-channel path works in integer units and hits the vertex exactly.
mass_gap = (symbol_masses(ref[0], CN) - symbol_masses(pc[0], CN)).abs().max()
assert mass_gap < 1e-5, f"masse per simbolo: max |d| = {mass_gap:.3e}"

same_facet = (ref[0][:, :2] == pc[0][:, :2]).all(dim=1)
for r, s, name in ((ref[1], pc[1], "beta_star"), (ref[3], pc[3], "beta_old")):
    d = (r[same_facet] - s[same_facet]).abs() / (r[same_facet].abs() + 1e-12)
    assert d.max() < 1e-4, f"{name} su faccia comune: max rel {d.max():.3e}"

what_ref = (w / ref[2]) / a_const
vertex_dist = (what_ref.unsqueeze(1) - Q.unsqueeze(0)).abs().min(dim=1).values
assert vertex_dist[~same_facet].max() < 1e-4, (
    f"facce diverse fuori da un vertice: {vertex_dist[~same_facet].max():.3e}"
)
print(
    f"[B] scala costante == chiamata per-tensore sui livelli reali\n"
    f"    y* max|d|={y_gap:.2e}  masse max|d|={mass_gap:.2e}\n"
    f"    {int((~same_facet).sum())}/{w.numel()} pesi esattamente su un vertice, "
    f"dove il subgradiente non e' unico (il paper lo prevede)"
)

if cp is None:
    print("[C/D] SALTATI: cvxpy non installato")
    sys.exit(0)

# ------------------------------------------------- C/D. cvxpy, livelli reali
M = 60
g2 = torch.Generator().manual_seed(11)
a_het = torch.exp(torch.rand(M, generator=g2) * 4.6 - 6.0)   # ~2 decadi
w_het = torch.randn(M, generator=g2) * (a_het * 4.0)
# The representable interval is [-8a, 7a] with y <= 1, so a weight must stay
# inside it or the inner problem is infeasible. The trainer guarantees this by
# clamping to [qn*a, qp*a] before calling the solver; the test does the same.
w_het = w_het.clamp(-a_het * 8.0, a_het * 7.0)
w_het[:5] = a_het[:5] * 6.98        # al pavimento di rappresentabilita'
w_het[5:10] = -a_het[5:10] * 7.98
w_het[10:14] = a_het[10:14] * 1e-4  # quasi zero
g3 = torch.Generator().manual_seed(3)
xi_h = torch.rand(CN, generator=g3) * 3e-7

x_pc, beta_pc, y_pc, bold_pc = K(xi_h, Q, w_het, CN, DEV, T1, T3, T2, scale=a_het)

xi_np = xi_h.numpy().astype(np.float64)
q_np = Q.numpy().astype(np.float64)


def phi(wi, ai):
    """Exact optimal value of the inner problem with REAL levels a_i*q."""
    v_i = ai * q_np
    x = cp.Variable(CN, nonneg=True)
    y = cp.Variable(nonneg=True)
    prob = cp.Problem(
        cp.Minimize(xi_np @ x + T1 * cp.quad_over_lin(wi, y) + T2 * y),
        [v_i @ x == wi, cp.sum(x) == y, y <= 1],
    )
    try:
        prob.solve(
            solver=cp.CLARABEL,
            tol_gap_abs=1e-14,
            tol_gap_rel=1e-14,
            tol_feas=1e-14,
        )
    except cp.error.SolverError:
        return None, None
    if not prob.status.startswith("optimal"):
        return None, None
    return float(prob.value), float(y.value)


worst_obj, worst_y, n_c = 0.0, 0.0, 0
for i in range(M):
    wi, ai = float(w_het[i]), float(a_het[i])
    ref_obj, ref_y = phi(wi, ai)
    if ref_obj is None:
        continue
    ys = float(y_pc[i])
    mass = symbol_masses(x_pc[i : i + 1], CN)[0].numpy().astype(np.float64)
    ours = xi_np @ mass + T1 * wi * wi / max(ys, 1e-30) + T2 * ys
    worst_obj = max(worst_obj, (ours - ref_obj) / max(abs(ref_obj), 1e-14))
    worst_y = max(worst_y, abs(ys - ref_y))
    n_c += 1

print(
    f"[C] cvxpy su {n_c} pesi, a in [{a_het.min():.2e}, {a_het.max():.2e}]\n"
    f"    eccesso relativo sull'ottimo : {worst_obj:.3e}\n"
    f"    |y* - y*_cvxpy|              : {worst_y:.3e}"
)
assert worst_obj < 1e-5, f"solver PEGGIORE dell'ottimo di {worst_obj:.2e}"
assert worst_y < 1e-3, f"y* discorda di {worst_y:.2e}"

# D. beta_star == dPhi/dw by central differences. The step is 1e-2*|w|: the
# error grows as h shrinks (8e-5 at 1e-2, 0.19 at 1e-5), which is solver noise
# amplified by 1/(2h), not truncation. Weights within 5e-2 of a hull vertex are
# skipped: there the derivative is genuinely one-sided.
worst_grad, n_d = 0.0, 0
for i in range(M):
    wi, ai = float(w_het[i]), float(a_het[i])
    whn = (wi / ai) / max(float(y_pc[i]), 1e-30)
    if float(np.abs(q_np - whn).min()) <= 5e-2 or wi == 0.0:
        continue
    h = 1e-2 * abs(wi)
    p1, _ = phi(wi + h, ai)
    p0, _ = phi(wi - h, ai)
    if p1 is None or p0 is None:
        continue
    fd = (p1 - p0) / (2 * h)
    worst_grad = max(
        worst_grad, abs(float(beta_pc[i]) - fd) / max(abs(fd), 1e-14)
    )
    n_d += 1

print(
    f"[D] beta_star vs dPhi/dw (differenze centrali) : {worst_grad:.3e} "
    f"su {n_d} pesi interni alla faccia"
)
assert worst_grad < 1e-3, f"beta_star discorda dalla derivata di {worst_grad:.2e}"

print("\nTUTTI I CONTROLLI SUPERATI")
