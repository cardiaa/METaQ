"""Offline check of the entropy dual's convergence rate. CPU, one second.

WHAT WENT WRONG IN test_232/233. entropy_coeff reaches the weights only through
xi: the per-weight subproblem is solved against the RAW bucket costs xi_b
(knapsack_perspective_leonardo says so explicitly, "entropy_coeff does NOT scale
the bucket costs"). So the entropy term is exactly as strong as the dual is
converged, and no stronger.

The dual ascends by xi += dual_step * g with g = (counts - c*)/upper_c in [-1,1]
(the test_134 normalization), so an ABSOLUTE dual_step moves xi a fixed amount
per iteration. But the range xi has to cover is

    xi_hi - xi_lo = entropy_coeff * log2(upper_c / lower_c)

which is proportional to entropy_coeff. Raise the coefficient and the target
recedes at the same rate: the dual converges slower, and the entropy term gets
WEAKER. Two runs measured it.

This script computes the dimensionless quantity that should have been held
fixed -- how many times per epoch the dual can traverse its own range -- for the
runs that worked and the runs that did not, and reports the --dual_step that
restores it in relative mode.
"""

import math

RUNS = [
    # label, entropy_coeff, dual_step, upper_c (params), steps/epoch, entropy_every
    ("test_168  ResNet-18, lossless", 3e-8, 3e-9, 11689512, 1251, 4),
    ("test_220  AlexNet, T3 alone",   1e-7, 3e-9, 61100840,  312, 4),
    ("test_232  EffNet, top 3e-7",    3e-7, 3e-9,  5288548,  312, 4),
    ("test_233  EffNet, top 3e-6",    3e-6, 3e-9,  5288548,  312, 4),
]
LOWER_C = 0.01
MAX_ITER = 3
# g = (counts - c*)/upper_c. counts sum to upper_c over C=16 buckets, c* is
# negligible while xi is far from its bound, so the typical g is about 1/C.
TYPICAL_G = 1.0 / 16.0


def xi_range(entropy_coeff, upper_c):
    return entropy_coeff * math.log2(upper_c / LOWER_C)


def traversals_per_epoch(entropy_coeff, dual_step, upper_c, steps, every):
    calls = steps / every
    travel = dual_step * TYPICAL_G * MAX_ITER * calls
    return travel / xi_range(entropy_coeff, upper_c)


def main():
    print(f"{'run':34s} {'xi range':>10s} {'travel/ep':>10s} {'traversals/ep':>14s}")
    ref = None
    for label, ec, ds, uc, st, ev in RUNS:
        r = xi_range(ec, uc)
        t = ds * TYPICAL_G * MAX_ITER * (st / ev)
        n = t / r
        if ref is None:
            ref = n
        print(f"{label:34s} {r:10.3e} {t:10.3e} {n:14.4f}")
    print()
    print(f"reference: test_168 traverses its dual range {ref:.2f} times per epoch")
    print("test_233 is slower than that by a factor of "
          f"{ref / traversals_per_epoch(3e-6, 3e-9, 5288548, 312, 4):.0f}.")
    print()
    print("--dual_step in RELATIVE mode (a fraction of the range per iteration)")
    print("that reproduces test_168's rate, for EfficientNet-B0 at 312 steps and")
    print("entropy_every 4, i.e. 78 dual calls per epoch:")
    calls = 312 / 4
    rel = ref / (TYPICAL_G * MAX_ITER * calls)
    print(f"    --dual_step {rel:.3e} --dual_step_mode relative")
    print("  and note it no longer depends on entropy_coeff, which is the point:")
    for ec in (3e-8, 3e-7, 3e-6):
        eff = rel * xi_range(ec, 5288548)
        got = traversals_per_epoch(ec, eff, 5288548, 312, 4)
        print(f"    entropy_coeff {ec:.0e} -> absolute step {eff:.3e}, "
              f"traversals/epoch {got:.2f}")
        assert abs(got - ref) < 1e-9
    print()
    print("OK")


if __name__ == "__main__":
    main()
