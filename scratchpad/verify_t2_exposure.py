"""Offline check of the T2 dose: schedule sums, duty cycle, cumulative exposure.

Runs on CPU in a second. Two things are verified and one is computed.

VERIFIED 1 -- the model of the calibration is exact. Re-deriving test_229's eight
T2_final values from its logged quantiles and step sizes reproduces the logged
numbers to the last digit, which is what licenses using the same arithmetic to
predict a run that has not happened.

VERIFIED 2 -- the duty-cycle fix does not touch runs with T3 off. With
entropy_coeff = 0 the sums are bit-identical to the pre-fix ones.

COMPUTED -- the exposure-matched --t2_scale for any (epoch count, schedule shape,
entropy_every). The quantity that transfers between runs is the CUMULATIVE
EXPOSURE T2 * S_linear, never the scale: the schedule sums absorb the epoch
count, the learning-rate shape and the duty cycle. test_229 measured the map
between exposure (in units of its own T2) and sparsity one epoch at a time:

    exposure   31.4 -> 42.4%    62.6 -> 51.0%    96.1 -> 57.3%   191.7 -> 70.6%

so a target sparsity picks an exposure and the exposure picks the scale.
"""

import math

# ---------------------------------------------------------------- fixed inputs
LR = 2e-3
STEPS_PER_EPOCH = 1251          # ImageNet, global batch 1024
DAMPING = 0.1                   # 1 - momentum, SGD 0.9
T1 = 1e-5

# From test_229's [LAYERWISE T2 CALIBRATION] line (epoch 1) and [LSQ GRID INIT].
# Both are deterministic given the checkpoint and --layer_C, so any AlexNet run
# on the frozen recipe calibrates against exactly these numbers.
LAYERS = ["conv1", "conv2", "conv3", "conv4", "conv5", "fc6", "fc7", "fc8"]
QUANTILES = [0.008097991347312927, 0.01713740825653076, 0.01794455386698246,
             0.01519688405096531, 0.014912177808582783, 0.00842646136879921,
             0.01027640886604786, 0.011361249722540379]
SCALES = [0.0032733455300331116, 0.0026350344996899366, 0.016431774944067,
          0.011198443360626698, 0.01033603772521019, 0.003223010338842869,
          0.0038493319880217314, 0.008425125852227211]
LEVELS = [256, 256, 16, 16, 16, 16, 16, 16]
T2_229 = [2.8795106442330715e-06, 7.63613444212975e-06, 1.8099772786566894e-06,
          1.2437743347795783e-06, 1.1706245100428156e-06, 2.614756001537991e-07,
          3.828942987472408e-07, 6.966420949065991e-07]

# test_229's own exposure/sparsity map, cumulative sums recomputed below.
EXPOSURE_AT_42_PERCENT = 31.42          # epoch 10 of test_229


def persp_lr(e, n_epochs, lr_warmup, phase, diag, ramp, floor):
    """Mirror of _persp_lr in utils/trainer_on_gpus_pretrained.py."""
    if lr_warmup > 0 and e < lr_warmup:
        return (e + 1) / float(lr_warmup + 1)
    if phase:
        if e < diag:
            return 1.0
        if ramp <= 0:
            return floor
        p = min(1.0, max(0.0, (e - diag + 1) / float(ramp)))
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * p))
    ramp_end = max(0, lr_warmup)
    decay_len = max(1, n_epochs - ramp_end)
    if e <= ramp_end:
        return 1.0
    p = min(1.0, (e - ramp_end) / decay_len)
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * p))


def schedule_sums(n_epochs, phase=False, diag=0, ramp=0, floor=0.01,
                  lr_warmup=1, entropy_every=1, entropy_on=False):
    """Mirror of _l1_schedule_sums, duty cycle included."""
    duty = 1.0 / float(max(1, entropy_every)) if entropy_on else 1.0
    linear = constant = 0.0
    for e in range(n_epochs):
        if phase:
            if e < diag:
                fraction = 0.0
            elif ramp > 0 and e < diag + ramp:
                fraction = (e - diag + 1) / float(ramp)
            else:
                fraction = 1.0
        else:
            fraction = 1.0
        if fraction <= 0.0:
            continue
        weight = LR * persp_lr(e, n_epochs, lr_warmup, phase, diag, ramp, floor)
        weight *= STEPS_PER_EPOCH * duty
        linear += weight * fraction
        constant += weight
    return linear / DAMPING, constant / DAMPING


def t2_final(idx, linear_sum, constant_sum, scale):
    """Mirror of _t2_from_displacement."""
    qn, qp = -(LEVELS[idx] // 2), LEVELS[idx] // 2 - 1
    reach = max(0.5 * (qp + abs(qn)) * SCALES[idx], 1e-12)
    distance = max(QUANTILES[idx] - SCALES[idx] / 2.0, 0.0)
    residual = distance - T1 * reach * constant_sum
    if residual <= 0.0 or linear_sum <= 0.0:
        return 0.0
    return residual * reach / linear_sum * scale


def main():
    print("=" * 72)
    print("1. reproduce test_229's calibration (60 ep, phase 0/40/20, min_lr 1e-4)")
    lin, con = schedule_sums(60, phase=True, ramp=40, floor=1e-4 / LR)
    print(f"   schedule_sums = ({lin:.6f}, {con:.6f})")
    print("   logged        = (191.66200331232554, 526.0821359472495)")
    worst = 0.0
    for i, name in enumerate(LAYERS):
        got = t2_final(i, lin, con, 0.31)
        worst = max(worst, abs(got / T2_229[i] - 1.0))
        print(f"     {name:6s} {got:.6e}  logged {T2_229[i]:.6e}")
    print(f"   worst relative error: {worst:.2e}")
    assert worst < 1e-9, "calibration model does not reproduce the logged run"

    print()
    print("2. duty cycle is inert when T3 is off")
    a = schedule_sums(60, entropy_every=4, entropy_on=False)
    b = schedule_sums(60, entropy_every=1, entropy_on=False)
    print(f"   entropy_every=4, T3 off: {a[0]:.6f}")
    print(f"   entropy_every=1, T3 off: {b[0]:.6f}")
    assert a == b, "the fix must not move a T2-only run"

    print()
    print("3. exposure-matched --t2_scale for a target of 42.4% sparsity")
    configs = [
        ("60 ep legacy, T3 on  (test_229bis)", dict(n_epochs=60, entropy_every=4, entropy_on=True)),
        ("60 ep legacy, T2 only", dict(n_epochs=60, entropy_every=1, entropy_on=False)),
        ("20 ep legacy, T3 on", dict(n_epochs=20, entropy_every=4, entropy_on=True)),
        ("20 ep legacy, T2 only", dict(n_epochs=20, entropy_every=1, entropy_on=False)),
    ]
    for label, kwargs in configs:
        lin, con = schedule_sums(**kwargs)
        scales = []
        for i in range(len(LAYERS)):
            unit = t2_final(i, lin, con, 1.0)          # T2_final at scale = 1
            # exposure delivered = T2_final * linear_sum, in units of T2_229
            scales.append(T2_229[i] * EXPOSURE_AT_42_PERCENT / (unit * lin))
        # fc6/fc7 carry 54.5M of AlexNet's 61.1M weights: they set the number
        print(f"   {label:36s} sums=({lin:7.2f},{con:7.2f})  "
              f"t2_scale={0.5 * (scales[5] + scales[6]):.4f}")

    print()
    print("   NOTE the 20-epoch rows are here only to show how far the scale "
          "moves\n   with the budget; 20 epochs cannot be lossless on this "
          "recipe (test_223).")
    print("=" * 72)
    print("OK")


if __name__ == "__main__":
    main()
