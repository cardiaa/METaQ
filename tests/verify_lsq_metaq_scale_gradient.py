"""Finite-difference check for the LSQ-METaQ scale envelope derivative.

This script intentionally uses only the Python standard library so it can run
on the local development machine as well as on Leonardo. It verifies regular
interior solutions; boundary/kink cases are detected and reported separately
because the value function has a set-valued subgradient there.
"""

from __future__ import annotations

import math
import random


def lower_hull(q, xi):
    hull = []
    for point in zip(q, xi):
        while len(hull) >= 2:
            q0, x0 = hull[-2]
            q1, x1 = hull[-1]
            q2, x2 = point
            left_slope = (x1 - x0) / (q1 - q0)
            right_slope = (x2 - x1) / (q2 - q1)
            if right_slope > left_slope:
                break
            hull.pop()
        hull.append(point)
    return hull


def segment_at(hull, target_q):
    tolerance = 1e-10
    for index, ((ql, xil), (qr, xir)) in enumerate(zip(hull, hull[1:])):
        if ql - tolerance <= target_q <= qr + tolerance:
            theta = (target_q - ql) / (qr - ql)
            value = xil + theta * (xir - xil)
            slope_q = (xir - xil) / (qr - ql)
            at_kink = (
                abs(target_q - ql) <= tolerance
                or abs(target_q - qr) <= tolerance
            )
            return index, value, slope_q, at_kink
    raise ValueError(f"target {target_q} is outside the codebook hull")


def solve_scalar(w, scale, q, xi, perspective_coeff, sparsity_coeff):
    hull = lower_hull(q, xi)
    q_min, q_max = hull[0][0], hull[-1][0]
    side = scale * (q_max if w >= 0 else abs(q_min))
    y_min = abs(w) / side
    if not 0 < y_min <= 1:
        raise ValueError("weight is not representable by the current scale")

    candidates = [y_min, 1.0]
    for q_breakpoint, _ in hull:
        value = w / (scale * q_breakpoint) if q_breakpoint != 0 else None
        if value is not None and y_min <= value <= 1.0:
            candidates.append(value)

    # On a hull segment, K(y)=beta*w+intercept*y. Add its stationary point.
    for (ql, xil), (qr, xir) in zip(hull, hull[1:]):
        slope_q = (xir - xil) / (qr - ql)
        beta = slope_q / scale
        intercept = xil - slope_q * ql
        denominator = intercept + sparsity_coeff
        if perspective_coeff > 0 and denominator > 0:
            y = abs(w) * math.sqrt(perspective_coeff / denominator)
            target_q = w / (scale * y) if y > 0 else math.inf
            if y_min <= y <= 1.0 and ql <= target_q <= qr:
                candidates.append(y)

    best = None
    for y in candidates:
        target_q = w / (scale * y)
        try:
            segment, hull_value, slope_q, at_kink = segment_at(hull, target_q)
        except ValueError:
            continue
        objective = y * hull_value + perspective_coeff * w * w / y + sparsity_coeff * y
        record = (objective, y, segment, slope_q, at_kink)
        if best is None or record[0] < best[0]:
            best = record
    if best is None:
        raise RuntimeError("no feasible candidate")

    objective, y, segment, slope_q, at_kink = best
    beta_old = slope_q / scale
    at_representation_floor = abs(y - y_min) < 1e-8
    if at_representation_floor:
        q_edge, xi_edge = hull[-1] if w >= 0 else hull[0]
        v_edge = scale * q_edge
        beta_old = (xi_edge + sparsity_coeff) / v_edge - perspective_coeff * v_edge
    predicted_scale_gradient = -beta_old * w / scale
    nonsmooth_kink = at_kink and not at_representation_floor
    return (
        objective,
        predicted_scale_gradient,
        nonsmooth_kink,
        at_representation_floor,
        y,
        segment,
    )


def finite_difference(w, scale, q, xi, perspective_coeff, sparsity_coeff):
    epsilon = 1e-5 * scale
    plus = solve_scalar(w, scale + epsilon, q, xi, perspective_coeff, sparsity_coeff)[0]
    minus = solve_scalar(w, scale - epsilon, q, xi, perspective_coeff, sparsity_coeff)[0]
    return (plus - minus) / (2 * epsilon)


def main():
    random.seed(7)
    q = [-8.0, -6.0, -3.0, -1.0, 1.0, 2.0, 4.0, 7.0]
    checked = 0
    skipped_kinks = 0
    checked_representation_boundaries = 0
    worst_relative_error = 0.0

    for _ in range(500):
        # Arbitrary dual costs are intentional: lower_hull removes dominated
        # buckets exactly as the METaQ inner solver does.
        xi = [random.uniform(-0.3, 0.8) for _ in q]
        scale = random.uniform(0.08, 0.4)
        w = random.uniform(-0.75, 0.75) * scale * 7.0
        if abs(w) < 1e-4:
            continue
        perspective_coeff = random.uniform(0.01, 0.2)
        sparsity_coeff = random.uniform(0.01, 0.3)

        try:
            value, predicted, nonsmooth_kink, representation_boundary, _, _ = solve_scalar(
                w, scale, q, xi, perspective_coeff, sparsity_coeff
            )
            observed = finite_difference(w, scale, q, xi, perspective_coeff, sparsity_coeff)
        except ValueError:
            continue

        if nonsmooth_kink:
            skipped_kinks += 1
            continue
        checked_representation_boundaries += int(representation_boundary)

        scale_error = max(1.0, abs(predicted), abs(observed), abs(value))
        relative_error = abs(predicted - observed) / scale_error
        worst_relative_error = max(worst_relative_error, relative_error)
        if relative_error > 2e-5:
            raise AssertionError(
                f"scale gradient mismatch: predicted={predicted}, "
                f"observed={observed}, relative_error={relative_error}"
            )
        checked += 1

    if checked < 25:
        raise AssertionError(f"too few regular cases checked: {checked}")
    if checked_representation_boundaries == 0:
        raise AssertionError("no representability-boundary case was verified")
    print(
        "LSQ-METaQ scale gradient verified: "
        f"regular_cases={checked}, "
        f"representability_boundaries={checked_representation_boundaries}, "
        f"nonsmooth_kinks_skipped={skipped_kinks}, "
        f"worst_relative_error={worst_relative_error:.3e}"
    )


if __name__ == "__main__":
    main()
