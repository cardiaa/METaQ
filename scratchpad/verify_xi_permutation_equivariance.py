"""Regression test for bucket-labelled entropy-dual multipliers.

The dual multiplier xi_b belongs to the fixed codebook symbol v_b.  A sign
reflection maps (v, xi, w) to (-reverse(v), reverse(xi), -w), which represents
the same optimization problem in reflected coordinates.  The perspective
FISTA update must therefore reflect xi and negate the weight gradient.  Sorting
xi after an update breaks this equivariance by reassigning costs to symbols.
"""

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.optimization import FISTA_perspective_leonardo


def main() -> None:
    device = torch.device("cpu")
    v = torch.tensor([-3.0, -1.0, 1.0, 3.0], device=device)
    w = torch.tensor([-2.4, -1.3, -0.4, 0.2, 1.1, 2.6], device=device)
    # Deliberately non-monotone: each entry is tied to its same-index symbol.
    xi = torch.tensor([-0.12, 0.16, -0.21, 0.05], device=device)

    kwargs = dict(
        C=4,
        upper_c=float(w.numel()),
        lower_c=1e-2,
        perspective_coeff=1e-2,
        entropy_coeff=1e-1,
        sparsity_coeff=1e-3,
        subgradient_step=1e5,
        device=device,
        max_iterations=4,
        dual_step=5e-2,
    )
    xi_out, beta, _ = FISTA_perspective_leonardo(xi, v, w, **kwargs)
    xi_reflected, beta_reflected, _ = FISTA_perspective_leonardo(
        xi.flip(0), -v.flip(0), -w, **kwargs
    )

    torch.testing.assert_close(xi_reflected, xi_out.flip(0), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(beta_reflected, -beta, rtol=1e-5, atol=1e-6)
    assert not torch.all(xi_out[:-1] <= xi_out[1:]), (
        "The test data no longer exercises a non-monotone multiplier update."
    )
    print("PASS: xi update is equivariant under codebook reflection.")


if __name__ == "__main__":
    main()
