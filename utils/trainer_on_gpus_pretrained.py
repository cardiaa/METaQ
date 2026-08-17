import torch
import os
import time 
import math
import numpy as np
import copy
import struct
import sys
import torch.optim as optim
import torch.distributed as dist
import gc
from utils.quantize_and_compress import compute_entropy, quantize_weights_center, compute_entropyGPU, quantize_weights_centerGPU, compute_entropy_hist
from utils.optimization import FISTA, FISTA_leonardo, FISTA_perspective_leonardo, FISTA_prox_leonardo, ProximalBM, test_accuracy, test_accuracyGPU
from utils.knapsack import prox_perspective_leonardo
from utils.knapsack import knapsack_specialized_pruning, knapsack_specialized_pruning_sparse_leonardo
from utils.weight_utils import initialize_weights
from utils.lsq import (
    expand_scale_flat,
    initial_weight_step_size,
    mse_weight_step_size,
    mse_weight_step_size_per_channel,
    metaq_scale_gradient,
    quantize_weight,
    signed_integer_codebook,
    task_scale_gradient,
)
from utils.quantize_and_compress import compress_zstd, BestQuantization, pack_bitmask, pack_bitmaskGPU
from datetime import datetime, timedelta


def _update_adiabatic_progress(
    progress,
    good_epochs,
    sparse_accuracy,
    target_accuracy,
    tolerance,
    step,
    backoff,
    patience,
):
    """Return the sparsity progress to use in the next epoch.

    Accuracy at or above ``target_accuracy`` accumulates patience and eventually
    advances compression. Accuracy below ``target_accuracy - tolerance`` rolls
    sparsity back. Values in the hysteresis band hold the current point so that
    validation noise does not make the controller chatter.
    """
    floor = target_accuracy - tolerance

    if sparse_accuracy >= target_accuracy:
        good_epochs += 1
        if progress >= 1.0:
            return 1.0, good_epochs, "complete"
        if good_epochs >= patience:
            return min(1.0, progress + step), 0, "advance"
        return progress, good_epochs, "patience"

    if sparse_accuracy < floor:
        next_progress = max(0.0, progress - backoff)
        action = "backoff" if next_progress < progress else "floor"
        return next_progress, 0, action

    return progress, 0, "hold"


def _aggregate_codebook_gradient(
    grad_flat: torch.Tensor,
    assignment: torch.Tensor,
    num_centroids: int,
) -> torch.Tensor:
    """Sum per-weight gradients into their fixed codebook buckets."""
    if grad_flat.ndim != 1 or assignment.ndim != 1:
        raise ValueError("Codebook gradients and assignments must be flat tensors.")
    if grad_flat.numel() != assignment.numel():
        raise ValueError("Codebook gradient and assignment sizes do not match.")

    centroid_grad = torch.zeros(
        num_centroids,
        dtype=grad_flat.dtype,
        device=grad_flat.device,
    )
    centroid_grad.scatter_add_(0, assignment, grad_flat)
    return centroid_grad


def _lloyd_codebook_1d(
    values_flat: torch.Tensor,
    initial_centroids: torch.Tensor,
    max_iterations: int,
    tolerance: float = 1e-7,
) -> tuple[torch.Tensor, torch.Tensor, int, float, float, float]:
    """Run scalar Lloyd clustering from a sorted linear initialization."""
    if values_flat.ndim != 1 or initial_centroids.ndim != 1:
        raise ValueError("Lloyd values and centroids must be flat tensors.")
    if initial_centroids.numel() < 2:
        raise ValueError("Lloyd clustering requires at least two centroids.")
    if max_iterations < 0:
        raise ValueError("Lloyd max_iterations must be >= 0.")

    centroids = torch.sort(initial_centroids.float()).values

    def assign(current_centroids):
        boundaries = (current_centroids[:-1] + current_centroids[1:]) / 2
        return torch.bucketize(
            values_flat,
            boundaries,
            right=False,
        ).clamp_(0, current_centroids.numel() - 1)

    assignment = assign(centroids)
    initial_error = values_flat.float() - centroids[assignment]
    initial_mse = initial_error.square().mean().item()
    iterations_run = 0
    final_shift = 0.0

    for iteration in range(max_iterations):
        counts = torch.bincount(
            assignment,
            minlength=centroids.numel(),
        )
        sums = torch.zeros_like(centroids)
        sums.scatter_add_(0, assignment, values_flat.float())

        updated = centroids.clone()
        nonempty = counts > 0
        updated[nonempty] = (
            sums[nonempty]
            / counts[nonempty].to(dtype=sums.dtype)
        )
        updated = torch.sort(updated).values

        final_shift = (updated - centroids).abs().max().item()
        centroids = updated
        assignment = assign(centroids)
        iterations_run = iteration + 1
        if final_shift <= tolerance:
            break

    final_error = values_flat.float() - centroids[assignment]
    final_mse = final_error.square().mean().item()
    return (
        centroids,
        assignment,
        iterations_run,
        initial_mse,
        final_mse,
        final_shift,
    )


def _rebuild_sorted_codebook(
    values_flat: torch.Tensor,
    assignment: torch.Tensor,
    old_centroids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Retie weights while preserving fixed cluster membership.

    Sorting keeps the codebook valid for bucketize-based evaluation. If two
    centroids cross, assignments are relabelled so each weight remains in the
    same cluster and retains the same centroid value.
    """
    num_centroids = old_centroids.numel()
    counts = torch.bincount(assignment, minlength=num_centroids)
    sums = torch.zeros_like(old_centroids)
    sums.scatter_add_(0, assignment, values_flat.to(dtype=old_centroids.dtype))
    updated = torch.where(
        counts > 0,
        sums / counts.clamp_min(1).to(dtype=sums.dtype),
        old_centroids,
    )

    sorted_centroids, old_index_at_new = torch.sort(updated)
    new_index_for_old = torch.empty_like(old_index_at_new)
    new_index_for_old[old_index_at_new] = torch.arange(
        num_centroids,
        dtype=assignment.dtype,
        device=assignment.device,
    )
    relabelled_assignment = new_index_for_old[assignment]
    projected_values = sorted_centroids[relabelled_assignment]
    return (
        sorted_centroids,
        relabelled_assignment,
        projected_values,
        new_index_for_old,
    )


def train_and_evaluate(model, model_name, criterion, C, lr, lambda_reg, alpha, perspective_coeff, entropy_coeff, subgradient_step, w0, r,
                       first_best_indices, BestQuantization_target_acc, final_target_acc, target_zstd_ratio, min_xi, max_xi, upper_c, 
                       lower_c, c1, c2, zeta, l, n_epochs, max_iterations, device, train_optimizer, entropy_optimizer, trainloader,
                       testloader, train_sampler, steps_per_epoch, delta, pruning, QuantizationType, sparsity_threshold, accuracy_tollerance,
                       gamma=1.0, metrics_interval=1, entropy_warmup_epochs=0, entropy_every=1, check_ddp_sync=False,
                       optimizer_weight_decay=None,
                       sparsity_coeff=0.0, mag_prune_ratio=0.5, use_perspective=False, target_sparsity=0.0,
                       sparsity_warmup_epochs=0, sparsity_ramp_power=1.0,
                       conv_sparsity=None, fc_sparsity=None, layer_sparsity=None,
                       flat_schedule=False, dual_step=0.5,
                       use_prox=False, prox_gamma=1e-7, prox_start_epoch=0,
                       sparsity_schedule=None, freeze_mask=False,
                       train_sparse=False, use_quantization=True,
                       quantizer="fixed", lsq_scale_lr=1e-5,
                       lsq_init="lsq", lsq_grad_scaling=True, lsq_per_channel=False,
                       distillation=False, distill_alpha=0.5, distill_tau=1.0,
                       min_lr=None, lsq_scale_lr_schedule=False,
                       lr_decay_epochs=None,
                       joint_lsq_metaq=False, bn_recalibration_batches=0,
                       layer_C=None,
                       train_centroids=False,
                       centroid_lr_scale=1.0, centroid_kmeans_iterations=0,
                       centroid_freeze_epoch=0,
                       adiabatic_accuracy_target=None, adiabatic_accuracy_tolerance=0.2,
                       adiabatic_step=0.02, adiabatic_backoff=0.04,
                       adiabatic_patience=2, evaluate_initial_model=False,
                       z_pruning=False):
    """Train and evaluate a model with optional entropy regularization.

    This function is intentionally self-contained because the compression
    experiments require tight control over the training loop.  The regular
    cross-entropy gradient is produced by `loss.backward()` and synchronized by
    DDP.  The entropy gradient is then added manually; therefore, when DDP is
    active, we explicitly all-reduce the final gradients before `optimizer.step()`.

    Args related to entropy:
        entropy_coeff: final weight of the entropy regularizer.  If zero, FISTA is
            never called.
        entropy_warmup_epochs: number of full epochs to train before enabling
            entropy regularization.
        entropy_every: apply entropy only every N global optimizer steps.
        max_iterations: number of FISTA/Proximal-BM iterations per entropy step.

    Args related to diagnostics:
        metrics_interval: run expensive validation/compression metrics every N
            epochs.
        check_ddp_sync: if true, log a checksum range across ranks to detect DDP
            divergence.
        flat_schedule: test_133.  Hold BOTH the learning rate and entropy_coeff constant for
            the whole run (no cosine tail, no exponential entropy_coeff ramp).  Tests 128/129/
            130 all shared one schedule SHAPE, scaled by entropy_coeff, so the cumulative
            "exposure" sum(entropy_coeff*lr) and its per-epoch RATE moved together and the
            collapse threshold could not be attributed to either one.  A flat
            schedule decouples them: the exposure accumulates slowly to a total
            well past the observed 1.6 wall while the per-epoch rate stays far
            below the rate of the run that survived (test_128, peak 0.40).
        dual_step: test_134.  Ascent step for the entropy dual, applied to the
            supergradient normalized by the layer size.  Replaces the old fixed
            1/subgradient_step = 1e-5 on a raw O(N) supergradient, which never
            converged (see FISTA_perspective_leonardo).
        use_prox: test_135.  Deliver phi as a PROXIMAL operator applied to the
            weights after the loss-only gradient step, instead of summing its
            subgradient into param.grad.  Same objective L + phi, different
            algorithm (proximal gradient).  Mutually exclusive with the beta*
            branch: when true, that branch is skipped entirely.
        prox_gamma: step of the proximal operator.  It is the knob that sets how
            far phi moves the weights, INDEPENDENTLY of the learning rate -- which
            is the whole point of the variant.  Calibrated offline on the real
            grid: at gamma=1e-5 one application already zeroes 3.4% of the
            weights and repeated application annihilates the layer within ~100
            calls, while 1e-7 gives ~11% zeroed over a full epoch of calls
            (worst case, with no loss pulling back).  Treat the first run as a
            calibration probe and read prox_diag from the log.
        prox_start_epoch: test_139.  First epoch (0-based) at which the proximal
            step runs.  It must be decoupled from entropy_warmup_epochs, which
            also gates the grid reset AND the start of the sparsity ramp: delaying
            that would delay pruning too.  Setting this late lets the sparsity
            ramp and the stabilisation run exactly as in the entropy_coeff=0 baseline, and
            only then turns the prox on to compress the values.  It also cuts the
            cost sharply, since a prox epoch costs ~5.6x a plain one (884s vs
            157s measured).  Default 0 = prox active as soon as entropy is.
        sparsity_schedule: test_143.  Iterative prune-and-heal schedule that
            REPLACES the smooth ramp (sparsity_warmup_epochs / ramp_power).  It
            is the cure for the instability found in test_141: pushing the big FC
            layers to extreme sparsity in one continuous ramp never lets them
            settle (accuracy oscillated 28-45), because the network chases a
            target that moves every epoch.  Here the per-layer target rises in
            STAGES, each held FLAT for several epochs so the net can actually
            converge at that level before the next increase -- Deep-Compression's
            prune-then-retrain, in miniature.
            Format: stages separated by ';', each "reach:hold:s1,...,s8" with
            1-based epochs -- `reach` is the epoch by which this stage's targets
            are fully reached (linearly, from the previous stage's vector or from
            0 for the first), and `hold` the epoch through which they are held.
            The first stage is reached with a gentle sub-ramp (a one-shot jump to
            85% collapsed the net in test_116); later stages are small steps and
            can jump in a single epoch.  When set, it takes priority over
            layer_sparsity / conv+fc / target_sparsity.
        freeze_mask: test_144.  Freeze the actual pruned INDEX SET (not just the
            threshold) for the duration of each plateau, Deep-Compression style.
            Today the mask is `|w| <= thr` recomputed every forward: as the net
            heals (weights move) a pruned weight can climb back above thr and a
            surviving one drop below it, so WHICH weights are pruned flip-flops
            epoch to epoch.  test_143 showed the dense net healing smoothly at the
            first plateau (A_NQ 30.9->34.8) while the DEPLOYED accuracy oscillated
            33-47 -- the instability was in the moving mask, not the training.
            With this flag the boolean mask is computed once when a new sparsity
            target is reached and held fixed while that target holds; the net then
            heals INTO a fixed sparse structure, and only refreezes when the
            target changes (a sub-ramp step or a stage jump).  Default False keeps
            the recompute-every-epoch behaviour of every earlier test.
        train_sparse: test_146.  Optimize the sparse SUBNETWORK directly instead
            of the dense net.  Every earlier run fake-quantized+pruned in the
            forward but restored the dense weights before optimizer.step(), so the
            pruned weights kept a non-zero gradient and drifted -- we minimized the
            loss over the DENSE net and deployed a masked copy, a mismatch.  With
            this flag the pruned positions are held at exactly zero and their
            gradient is zeroed, so only the surviving weights are updated: the
            objective becomes the sparse network we actually deploy.  This is the
            main untested methodological difference from Deep Compression (which
            trains the fixed sparse subnet).  Default False keeps the old
            behaviour.
        use_quantization: when false, train and evaluate the magnitude-pruned
            network in FP32. Compression metrics then count the nonzero values as
            raw FP32 rather than incorrectly counting quantization indices. This
            provides the pruning-first control used by test_155.
        layer_C: optional number of non-zero quantization levels for every weight
            tensor. This supports the Deep-Compression control (256 levels / 8
            bits for convolutional tensors and 32 levels / 5 bits for fully
            connected tensors) instead of forcing one global C.
        train_centroids: Deep-Compression-style codebook fine-tuning. After the
            adaptive grid is built, assignments are frozen, per-weight gradients
            are summed by bucket, and the shared centroid values are re-tied
            after every optimizer step. The initial implementation is restricted
            to the no-pruning, no-regularization control.
        centroid_lr_scale: multiplier applied to the summed centroid gradients
            before SGD. It is an explicit centroid learning-rate ratio relative
            to the base model learning rate.
        centroid_kmeans_iterations: number of scalar Lloyd iterations run from
            the linear grid before assignments are frozen. Zero reproduces the
            direct linear assignment used by tests 150-152.
        centroid_freeze_epoch: optional 1-based epoch at which dynamic QAT is
            converted into a fixed shared codebook. Zero freezes immediately
            when the adaptive grid is built, reproducing tests 150-153.
        adiabatic_accuracy_target: when set, replace the time-driven sparsity ramp
            with closed-loop control. A scalar progress multiplies the requested
            per-layer final targets. It advances only after
            ``adiabatic_patience`` evaluations at the target accuracy, holds
            inside the tolerance band, and backs off below the band.
    """

    torch.set_num_threads(1)

    local_rank = dist.get_rank() if dist.is_initialized() else 0

    if device.type == "cuda":
        torch.cuda.set_device(device.index if device.index is not None else local_rank)

    # Under the perspective reformulation the ridge is perspective_coeff * w^2 / y* (perspective
    # form), whose gradient 2*perspective_coeff*w/y* is applied EXPLICITLY per step.  The plain
    # SGD weight_decay (which would add a second, standard perspective_coeff*w ridge) is disabled
    # to avoid double-counting.
    wd_init = (
        float(optimizer_weight_decay)
        if optimizer_weight_decay is not None
        else (0.0 if use_perspective else perspective_coeff)
    )

    if train_optimizer == 'ADAM':
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=wd_init)
    elif train_optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd_init)
    else:
        raise ValueError(f"Unsupported optimizer: {train_optimizer}")

    # test_124: LR schedule tailored to the sparsity ramp.  During the ramp the
    # network must keep adapting, so the LR stays CONSTANT (test_118/119 showed a
    # decay there collapses the accuracy).  But once the ramp reaches full
    # sparsity, a constant LR is too hot at extreme sparsity (~89%): the few
    # surviving weights oscillate and the accuracy sbanda (test_123).  So we add a
    # cosine decay in the TAIL only, after the ramp completes.
    if use_perspective:
        _ramp_end = entropy_warmup_epochs + max(0, sparsity_warmup_epochs)   # epoch (0-based) at full sparsity
        _schedule_end = (
            centroid_freeze_epoch - 1
            if train_centroids and centroid_freeze_epoch > 0
            else n_epochs
        )
        _decay_len = max(1, _schedule_end - _ramp_end)
        # lr_decay_epochs decouples the length of the cosine descent from the
        # number of epochs. Left unset, the cosine spans the whole run as before.
        # Set to K, the rate reaches its floor after K epochs of descent and is
        # then HELD at the floor for the rest of the run. This separates the
        # compression phase (the descent, where PEAQ exposure is high) from a
        # healing phase (the floor tail, where PEAQ is weak and the network
        # settles into the compressed configuration). Test_191's tail showed
        # accuracy recovering about 0.05 per epoch at the floor rate, so a long
        # floor tail may recover the accuracy that a bare 25-epoch run leaves on
        # the table. Setting it equal to test_191's descent length makes the
        # first 25 epochs identical to test_191 and the rest a controlled tail.
        if lr_decay_epochs is not None:
            if lr_decay_epochs < 1:
                raise ValueError("--lr_decay_epochs must be >= 1.")
            _decay_len = int(lr_decay_epochs)
        # Floor of the cosine, as a fraction of the base rate. The historical
        # value is 1%, which for the 2e-5 used on DeiT means 2e-7: the tail is
        # effectively dead, and since PEAQ enters by being added to the loss
        # gradient it switches off along with it. Test_186 shows exactly that,
        # with compression bottoming at 8.18% while the rate is still 7.5e-6 and
        # then REVERSING to 8.27% as the rate collapses twelvefold over the last
        # eight epochs. Q-ViT instead floors its cosine at 1e-5 against a 2e-4
        # base, five times higher in relative terms, and holds it there for a
        # ten-epoch cooldown. --min_lr makes the floor an explicit absolute rate;
        # leaving it unset preserves the 1% behaviour exactly.
        _lr_floor = 0.01
        if min_lr is not None:
            if min_lr < 0:
                raise ValueError("--min_lr must be >= 0.")
            if min_lr >= lr:
                raise ValueError(
                    f"--min_lr ({min_lr}) must be below the base learning rate "
                    f"({lr})."
                )
            _lr_floor = min_lr / lr

        def _persp_lr(e):
            # test_133: flat_schedule holds the LR constant for the whole run, so
            # that the entropy displacement per epoch (proportional to entropy_coeff*lr) stays
            # flat and the cumulative exposure can be decoupled from its rate.
            if flat_schedule:
                return 1.0
            if e <= _ramp_end:
                return 1.0
            p = min(1.0, (e - _ramp_end) / _decay_len)
            return _lr_floor + (1.0 - _lr_floor) * 0.5 * (1.0 + math.cos(math.pi * p))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_persp_lr)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,
            gamma=0.1
        )

    # Per-parameter-tensor quantization state.
    # Each layer/tensor gets its own quantization grid v, xi, and optionally C.
    # All parameters are still needed to rebuild the full flat model.
    all_named_params = list(model.named_parameters())
    all_params = [param for _, param in all_named_params]

    # Quantize actual convolutional/linear weight tensors, not every tensor with
    # more than one dimension. The distinction is essential for transformers:
    # DeiT's positional embedding and class token are learned tensors but are not
    # matrix weights and must remain in floating point.
    quant_param_indices = [
        idx
        for idx, (name, param) in enumerate(all_named_params)
        if name.endswith(".weight") and param.ndim in (2, 4)
    ]
    params_for_quant = [all_params[idx] for idx in quant_param_indices]
    quant_param_names = [all_named_params[idx][0] for idx in quant_param_indices]
    quant_param_ids = {id(param) for param in params_for_quant}
    num_param_tensors = len(params_for_quant)
    if num_param_tensors == 0:
        raise ValueError(f"{model_name} has no convolutional or linear weights to quantize.")

    if layer_C is None:
        C_by_layer = [int(C)] * num_param_tensors
    else:
        C_by_layer = [int(c) for c in layer_C]
        if len(C_by_layer) != num_param_tensors:
            raise ValueError(
                f"--layer_C has {len(C_by_layer)} values but the model has "
                f"{num_param_tensors} quantized tensors."
            )
    if any(c < 2 for c in C_by_layer):
        raise ValueError("Every quantization level count in C/layer_C must be >= 2.")
    max_C = max(C_by_layer)
    lsq_enabled = quantizer == "lsq"
    if quantizer not in ("fixed", "lsq"):
        raise ValueError(f"Unsupported quantizer: {quantizer}")
    if lsq_enabled and not use_quantization:
        raise ValueError("--quantizer lsq requires --quantization Y.")
    if lsq_enabled and train_centroids:
        raise ValueError("LSQ and --train_centroids Y are mutually exclusive.")
    if lsq_enabled and not use_perspective:
        raise ValueError(
            "The LSQ implementation currently requires --perspective Y so zero "
            "is represented consistently as METaQ missing mass."
        )
    if lsq_enabled and use_prox:
        raise ValueError(
            "The LSQ implementation currently supports only the gradient path, "
            "not the proximal path."
        )
    if lsq_init not in ("lsq", "mse"):
        raise ValueError(f"Unsupported LSQ initialization: {lsq_init}")
    if bn_recalibration_batches < 0:
        raise ValueError("bn_recalibration_batches must be >= 0.")
    if joint_lsq_metaq and not lsq_enabled:
        raise ValueError("Joint LSQ-METaQ requires the LSQ quantizer.")
    if joint_lsq_metaq and not use_perspective:
        raise ValueError("Joint LSQ-METaQ requires --perspective Y.")
    if lsq_per_channel and not lsq_enabled:
        raise ValueError("--lsq_per_channel Y requires --quantizer lsq.")
    if lsq_per_channel and (
        mag_prune_ratio or target_sparsity or layer_sparsity
        or conv_sparsity is not None or fc_sparsity is not None
        or sparsity_schedule
    ):
        # The magnitude-pruning helpers derive their thresholds from a single
        # per-tensor level vector. Feeding them per-channel step sizes would
        # silently compare weights against the wrong levels, so the combination
        # is refused rather than approximated. Sparsity under per-channel comes
        # from the LSQ zero bin, exactly as in tests 168 to 175.
        raise ValueError(
            "--lsq_per_channel Y is wired for the zero-bin sparsity regime "
            "only: mag_prune_ratio, target_sparsity, conv/fc/layer_sparsity and "
            "sparsity_schedule must all be unset."
        )
    if lsq_per_channel and (use_prox or train_centroids):
        raise ValueError(
            "--lsq_per_channel Y supports neither the proximal path nor "
            "trained centroids."
        )
    if joint_lsq_metaq and use_prox:
        raise ValueError(
            "Joint LSQ-METaQ scale gradients are defined for the gradient path, "
            "not the proximal path."
        )

    lsq_integer_codebooks = None
    lsq_nonzero_integer_codebooks = None
    lsq_scales = None
    scale_optimizer = None
    if lsq_enabled:
        lsq_integer_codebooks = [
            signed_integer_codebook(C_layer, device)
            for C_layer in C_by_layer
        ]
        lsq_nonzero_integer_codebooks = [
            q[q != 0] for q in lsq_integer_codebooks
        ]
        initial_scales = []
        for param, q in zip(params_for_quant, lsq_integer_codebooks):
            if lsq_per_channel:
                if lsq_init == "mse":
                    value = mse_weight_step_size_per_channel(param, q)
                else:
                    rows = param.detach().float().reshape(param.shape[0], -1)
                    value = (
                        2.0 * rows.abs().mean(dim=1)
                        / math.sqrt(float(q[-1].item()))
                    ).clamp_min(1e-12)
                tensor_value = value.detach().to(
                    dtype=torch.float32, device=device
                ).clone()
            else:
                if lsq_init == "mse":
                    value = mse_weight_step_size(param, q)
                else:
                    value = initial_weight_step_size(param, float(q[-1].item()))
                tensor_value = torch.tensor(
                    value, dtype=torch.float32, device=device
                )
            initial_scales.append(torch.nn.Parameter(tensor_value))
        lsq_scales = initial_scales
        scale_optimizer = optim.Adam(
            lsq_scales,
            lr=lsq_scale_lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
        )

    # Giving the step sizes their own optimizer decoupled them from the weight
    # schedule, which was never intended. The weights decay along the cosine to
    # 1% of the base rate while the step sizes keep learning at a constant
    # lsq_scale_lr, so their relative speed changes by a factor of a hundred
    # across a run: on DeiT they start at half the weight rate and end at fifty
    # times it. Since the deployed model is a discontinuous function of the
    # latent weights, a grid that keeps moving after the network has frozen
    # reassigns weights between levels and shifts the zero-bin boundary, which
    # shows up as epoch-to-epoch accuracy jitter that never settles and as the
    # compression reversal of test_186, where the ratio bottomed at 8.18% and
    # climbed back to 8.27% while sparsity fell and the weight rate collapsed.
    # Canonical LSQ has no such split, since alpha sits in the weight optimizer,
    # and neither does Q-ViT, where alpha is an ordinary Parameter inside LAMB.
    # The default keeps the historical constant rate.
    scale_scheduler = None
    if scale_optimizer is not None and lsq_scale_lr_schedule:
        if not use_perspective:
            raise ValueError(
                "--lsq_scale_lr_schedule Y mirrors the perspective schedule and "
                "requires --perspective Y."
            )
        scale_scheduler = torch.optim.lr_scheduler.LambdaLR(
            scale_optimizer, lr_lambda=_persp_lr
        )

    # C_by_layer is the deployment alphabet size (including zero under LSQ).
    # METaQ treats zero as missing mass z=1-y, so its bucket problem receives
    # only the C-1 non-zero integer levels.
    metaq_C_by_layer = (
        [c - 1 for c in C_by_layer] if lsq_enabled else list(C_by_layer)
    )

    if local_rank == 0:
        total_model_params = sum(param.numel() for param in all_params)
        total_quant_params = sum(param.numel() for param in params_for_quant)
        print(
            "[QUANTIZED TENSORS] "
            f"count={num_param_tensors}, parameters={total_quant_params}, "
            f"fraction_of_model={total_quant_params / max(1, total_model_params):.6%}",
            flush=True,
        )
        for idx, (name, param, levels) in enumerate(
            zip(quant_param_names, params_for_quant, C_by_layer)
        ):
            print(
                f"[QUANTIZED TENSOR {idx}] name={name}, shape={tuple(param.shape)}, "
                f"parameters={param.numel()}, C={levels}",
                flush=True,
            )

    if not use_quantization:
        if not use_perspective:
            raise ValueError(
                "--quantization N currently supports magnitude pruning only; "
                "use it with --perspective Y."
            )
        if entropy_coeff != 0.0:
            raise ValueError(
                "--quantization N requires --entropy_coeff 0 because the entropy term is "
                "defined on quantization buckets."
            )
        if train_centroids:
            raise ValueError(
                "--train_centroids Y requires --quantization Y."
            )

    if train_centroids:
        if train_optimizer != "SGD":
            raise ValueError("--train_centroids Y currently requires SGD.")
        if not 0.0 < centroid_lr_scale <= 1.0:
            raise ValueError("--centroid_lr_scale must be in (0, 1].")
        if centroid_kmeans_iterations < 0:
            raise ValueError("--centroid_kmeans_iterations must be >= 0.")
        if centroid_freeze_epoch < 0 or centroid_freeze_epoch > n_epochs:
            raise ValueError(
                "--centroid_freeze_epoch must be 0 or a 1-based epoch within the run."
            )
        if (
            centroid_freeze_epoch > 0
            and centroid_freeze_epoch <= entropy_warmup_epochs
        ):
            raise ValueError(
                "--centroid_freeze_epoch must be after grid adaptation."
            )
        compression_terms_active = any(
            value is not None and float(value) != 0.0
            for value in (
                perspective_coeff,
                entropy_coeff,
                sparsity_coeff,
                mag_prune_ratio,
                target_sparsity,
                conv_sparsity,
                fc_sparsity,
            )
        )
        layer_pruning_active = (
            layer_sparsity is not None
            and any(float(value) != 0.0 for value in layer_sparsity)
        )
        if compression_terms_active or layer_pruning_active:
            raise ValueError(
                "--train_centroids Y currently requires perspective_coeff=entropy_coeff=sparsity_coeff=0 and no "
                "magnitude/per-layer sparsity."
            )
        if any(
            (
                use_prox,
                train_sparse,
                freeze_mask,
                sparsity_schedule is not None,
                adiabatic_accuracy_target is not None,
            )
        ):
            raise ValueError(
                "--train_centroids Y is currently a pure quantization control "
                "and cannot be combined with pruning, proximal, or adiabatic modes."
            )

    min_w, max_w = w0 - r, w0 + r
    if lsq_enabled:
        # Per-channel has no single level vector: v_list then holds the INTEGER
        # codebook and the step sizes reach the solver separately. Kept in sync
        # with _refresh_lsq_metaq_grids_, which rebuilds the same object.
        v_list = [
            q_nonzero.clone()
            if lsq_per_channel
            else scale.detach() * q_nonzero
            for scale, q_nonzero in zip(
                lsq_scales, lsq_nonzero_integer_codebooks
            )
        ]
    else:
        v_list = [
            torch.linspace(
                min_w,
                max_w - (max_w - min_w) / C_layer,
                steps=C_layer,
                device=device,
            )
            for C_layer in C_by_layer
        ]

    xi_list = []
    for C_layer in metaq_C_by_layer:
        # xi_layer[0] is the multiplier for the explicit zero/pruning symbol.
        # xi_layer[1:] are the multipliers for this layer's non-zero buckets.
        xi_zero = min_xi + (max_xi - min_xi) * torch.rand(1, device=device)

        xi_buckets = min_xi + (max_xi - min_xi) * torch.rand(C_layer, device=device)

        xi_layer = torch.cat([xi_zero, xi_buckets])

        if dist.is_initialized():
            dist.broadcast(xi_layer, src=0)

        xi_list.append(xi_layer)

    # Knowledge distillation: a frozen full-precision copy of the network taken
    # BEFORE any training, used as the teacher. Every DeiT run so far shows the
    # same shape, a peak within five to seven epochs and then a drift downwards,
    # because nothing anchors the student to the function we are trying to
    # preserve. The teacher is exactly that anchor. It is a separate module, so
    # the fake-quantization that temporarily overwrites the student's weights
    # during the forward pass leaves it untouched, and it never receives
    # gradients.
    teacher_model = None
    if distillation:
        source = model.module if hasattr(model, "module") else model
        teacher_model = copy.deepcopy(source).to(device)
        teacher_model.eval()
        for teacher_param in teacher_model.parameters():
            teacher_param.requires_grad_(False)
        if local_rank == 0:
            print(
                f"[DISTILLATION] teacher cloned from the pretrained weights, "
                f"alpha={distill_alpha}, tau={distill_tau}, "
                f"parameters={sum(p.numel() for p in teacher_model.parameters())}",
                flush=True,
            )

    log = ""
    accuracy = None
    accuracies, entropies, zstd_ratios = [], [], []
    distill_loss_last = None
    task_loss_last = None
    # test_126: track the SPARSE metrics we actually optimize (quantized+pruned
    # accuracy and the sparse compression ratio), not just the dense A_NQ / zstd.
    sparse_accuracies, sparse_ratios = [], []
    global_step = 0

    # The adaptive grid must be reset only once for the whole run, not once per
    # epoch.  It is triggered at the first real entropy step, so it remains
    # correct even when `global_step % entropy_every != 0` at batch 0.
    grid_reset_done = False
    fixed_codebook_assignments = [None] * num_param_tensors
    codebook_active = False

    # Fraction of each layer range used as explicit zero dead-zone.
    # If |w| <= deadzone_ratio * r_layer, the weight is quantized to exactly zero.
    deadzone_ratio = 0.0

    # Diagnostic mode for dual-zero pruning in evaluation/compression.
    # The pruning budget is taken from the original gamma/deadzone rule.
    #
    # Candidate weights are restricted to:
    #
    #     |w_i| <= dual_zero_candidate_multiplier * tau_gamma
    #
    # Among those candidates, weights are ranked by:
    #
    #     score_i = z_i / (|w_i| / tau_gamma + eps)^p
    #
    # This prevents the dual-zero score from pruning large-magnitude weights.
    dual_zero_rounding = "topk_gamma_budget_capped_z_over_abs_power"
    dual_zero_score_eps = 1e-6
    dual_zero_abs_power = 2.0
    dual_zero_candidate_multiplier = 1.25

    # DIAGNOSTIC (test_104): when False, training keeps the FP32 weights in the
    # forward/backward pass (no fake-quantized + deadzone-pruned STE-QAT).
    # This follows the thesis recipe: train FP32 with CE + L2 + entropy
    # subgradient, and quantize/prune only post-hoc for evaluation/compression.
    # test_108: quantization-aware training re-enabled so the loss SEES the
    # quantized weights and protects A_Q (which collapsed under FP32-only +
    # entropy in test_107).
    # test_110: the fake-quant forward also prunes (see prune_mode below) so the
    # loss sees the quantized + PRUNED model and learns to tolerate the pruning
    # (sparse_accuracy was stuck at ~3% because the net was not pruning-aware).
    # test_155: the public quantization switch makes this a genuine pruning-only
    # run. Magnitude masks are still applied below, but surviving values remain
    # FP32 in both the forward pass and deployment metrics.

    # ENTROPY SUBGRADIENT SANITIZATION (test_105).
    # The per-weight entropy subgradient is beta = xi_eff / v, which explodes
    # (and flips sign) for weights mapped to near-zero buckets (|v| ~ gap).
    # We winsorize beta per-coordinate to beta_clip_k * median(|beta|) of the
    # layer so the entropy signal is coherent rather than spike-dominated.
    beta_clip_k = 5.0

    # Fixed per-layer reference loss-gradient norm for entropy auto-scaling
    # (test_107).  The entropy update is scaled to entropy_coeff_current * ref_grad_norm.
    # Using a FIXED reference (captured at the first entropy step) instead of the
    # live loss-grad norm prevents a runaway: otherwise a dropping accuracy
    # inflates the loss gradient, which inflates the entropy push, which drops
    # accuracy further (death spiral observed in test_106 at entropy_coeff=5).
    entropy_ref_grad_norm = [None] * num_param_tensors

    # test_110: OPTIMIZATION-DRIVEN pruning.  Instead of the hand-coded magnitude
    # deadzone, we prune the weights for which the relaxed knapsack assigns most
    # of the mass to the zero/pruning symbol, i.e.
    #     z_i = 1 - sum_b x_{i,b} > z_prune_threshold.
    # delta therefore governs BOTH how many and which weights are pruned, through
    # the optimization itself (not a magnitude rule).  The per-layer masks are
    # computed in the dual block (from the current xi) and cached for the
    # fake-quant forward and for evaluation, so train and eval prune the same set.
    # test_113: perspective reformulation.  Under use_perspective the pruning is
    # magnitude-based (Frangioni's advice), driven by the perspective ridge L1
    # push; the entropy dual (FISTA/knapsack, the z-symbol, delta) is NOT used.
    # "magnitude" therefore disables the z-branch and the dual altogether.
    prune_mode = "z" if z_pruning else ("magnitude" if use_perspective else "z")
    z_prune_threshold = 0.5
    z_prune_masks = [None] * num_param_tensors
    # Per-layer magnitude thresholds for target_sparsity mode, frozen once per
    # epoch (stable + cheap) and reused by both the forward and evaluation.
    target_prune_thresholds = [None] * num_param_tensors
    # test_136: the effective per-layer sparsity target the frozen threshold was
    # computed for.  In prox mode the operator keeps creating exact zeros DURING
    # the epoch, so a threshold frozen at epoch start is stale by evaluation time;
    # keeping the target lets us recompute it on the spot (see _refresh_prune_thresholds).
    effective_ts_by_layer = [None] * num_param_tensors
    # test_144: the frozen pruned index set (boolean, flattened) per layer, and
    # the effective target it was built for -- so it can be refrozen only when the
    # target actually changes.  Used only when freeze_mask is on.
    frozen_prune_masks = [None] * num_param_tensors
    frozen_mask_ts = [None] * num_param_tensors

    # test_124: full Deep-Compression-style per-layer sparsity.  layer_sparsity is
    # a list with one target per quantized tensor (in the order of params_for_quant:
    # conv1, conv2, ..., fc6, fc7, fc8 for AlexNet).  It overrides conv/fc groups
    # and target_sparsity.
    if layer_sparsity is not None:
        if len(layer_sparsity) != num_param_tensors:
            raise ValueError(
                f"--layer_sparsity has {len(layer_sparsity)} values but the model "
                f"has {num_param_tensors} quantized tensors."
            )
        if local_rank == 0:
            print(f"layer_sparsity (per quantized tensor, in order) = {list(layer_sparsity)}", flush=True)

    if local_rank == 0 and layer_C is not None:
        print(f"layer_C (per quantized tensor, in order) = {C_by_layer}", flush=True)

    adiabatic_enabled = adiabatic_accuracy_target is not None
    if adiabatic_enabled:
        if not use_perspective:
            raise ValueError("Adiabatic sparsity control requires --perspective Y.")
        if train_sparse:
            raise ValueError(
                "Adiabatic rollback is incompatible with --train_sparse Y because "
                "hard-zeroed weights cannot be restored when sparsity backs off."
            )
        if sparsity_schedule:
            raise ValueError(
                "--adiabatic_accuracy_target and --sparsity_schedule are mutually exclusive."
            )
        if adiabatic_accuracy_tolerance < 0:
            raise ValueError("--adiabatic_accuracy_tolerance must be >= 0.")
        if not 0 < adiabatic_step <= 1:
            raise ValueError("--adiabatic_step must be in (0, 1].")
        if not 0 < adiabatic_backoff <= 1:
            raise ValueError("--adiabatic_backoff must be in (0, 1].")
        if adiabatic_patience < 1:
            raise ValueError("--adiabatic_patience must be >= 1.")

        if layer_sparsity is not None:
            adiabatic_base_targets = list(layer_sparsity)
        elif conv_sparsity is not None and fc_sparsity is not None:
            adiabatic_base_targets = [
                conv_sparsity if param.dim() == 4 else fc_sparsity
                for param in params_for_quant
            ]
        elif target_sparsity and target_sparsity > 0:
            adiabatic_base_targets = [target_sparsity] * num_param_tensors
        else:
            raise ValueError(
                "Adiabatic control needs final sparsity targets via --layer_sparsity, "
                "--conv_sparsity/--fc_sparsity, or --target_sparsity."
            )
        if any(not 0.0 <= target < 1.0 for target in adiabatic_base_targets):
            raise ValueError("Every adiabatic sparsity target must be in [0, 1).")

        adiabatic_progress = 0.0
        adiabatic_good_epochs = 0
        if local_rank == 0:
            print(
                f"[ADIABATIC CONFIG] target_accuracy={adiabatic_accuracy_target:.4f}, "
                f"floor={adiabatic_accuracy_target - adiabatic_accuracy_tolerance:.4f}, "
                f"step={adiabatic_step:.4f}, backoff={adiabatic_backoff:.4f}, "
                f"patience={adiabatic_patience}, base_targets={adiabatic_base_targets}",
                flush=True,
            )
    else:
        adiabatic_base_targets = None
        adiabatic_progress = 0.0
        adiabatic_good_epochs = 0
    adiabatic_best_target = None
    adiabatic_best_floor = None

    # NCCL barriers need the CUDA device id on some multi-node launches.
    def _dist_barrier():
        if dist.is_initialized():
            if dist.get_backend() == "nccl" and device.type == "cuda":
                dist.barrier(device_ids=[torch.cuda.current_device()])
            else:
                dist.barrier()

    # Temporarily overwrite model parameters with a flat vector.
    # Used to evaluate quantized/sparse weights without creating a deepcopy of
    # the whole DDP model.
    def _load_flat_params_(flat: torch.Tensor):
        offset = 0
        for p_ in model.parameters():
            n = p_.numel()
            p_.data.copy_(flat[offset:offset + n].view_as(p_))
            offset += n

    def _current_quant_levels(q_state):
        if lsq_enabled:
            if lsq_per_channel:
                # (O, C): one real level vector per output channel. Only the
                # non-LSQ deadzone and centroid paths consume this, and both are
                # refused under per-channel, so it exists for diagnostics only.
                return (
                    lsq_scales[q_state].detach().clamp_min(1e-12).unsqueeze(1)
                    * lsq_integer_codebooks[q_state].unsqueeze(0)
                )
            return (
                lsq_scales[q_state].detach().clamp_min(1e-12)
                * lsq_integer_codebooks[q_state]
            )
        return v_list[q_state]

    def _refresh_lsq_metaq_grids_():
        if not lsq_enabled:
            return
        with torch.no_grad():
            for q_state in range(num_param_tensors):
                if lsq_per_channel:
                    # There is no single level vector any more: each output
                    # channel has its own. The METaQ solver instead receives the
                    # INTEGER codebook plus a per-weight step size, which is
                    # exact because the lower convex envelope is invariant under
                    # positive rescaling of the abscissa.
                    v_list[q_state] = lsq_nonzero_integer_codebooks[q_state]
                else:
                    v_list[q_state] = (
                        lsq_scales[q_state].detach().clamp_min(1e-12)
                        * lsq_nonzero_integer_codebooks[q_state]
                    )

    def _metaq_scale_flat(p_idx, param):
        """Per-weight step sizes for the METaQ solver, or None per-tensor.

        Output channels are the leading dimension, so the flat layout repeats
        each channel's step size ``numel // O`` times.
        """
        if not (lsq_enabled and lsq_per_channel):
            return None
        return expand_scale_flat(
            lsq_scales[p_idx].detach().clamp_min(1e-12), param
        )

    def _add_clipping_scale_term(
        scale_grad, beta_star, clipping_direction, scale_param
    ):
        """Add the clipped-weight term of dphi/ds from Eq. (34).

        Outside the representable interval dclip/ds is q_n or q_p, so the term
        is sum_i beta*_i (dclip/ds)_i. With one step size per output channel
        that sum runs over each channel separately.
        """
        term = beta_star.float() * clipping_direction.float()
        if lsq_per_channel:
            return scale_grad + term.reshape(scale_param.numel(), -1).sum(dim=1)
        return scale_grad + term.sum()

    def _fake_quantize_weights_for_forward_():
        """
        Temporarily replaces quantized weight tensors with their per-layer
        fake-quantized version for the forward/backward pass.

        Bias tensors are not touched.

        Returns:
            backups: list of (parameter, original_float_tensor)
        """
        backups = []

        if not grid_reset_done:
            return backups

        with torch.no_grad():
            for q_state, param in enumerate(params_for_quant):
                original = param.data.clone()
                backups.append((param, original))

                v_layer = _current_quant_levels(q_state)
                w_flat = param.data.reshape(-1)

                fixed_assignment = fixed_codebook_assignments[q_state]
                if not use_quantization:
                    q_flat = w_flat.clone()
                elif train_centroids and fixed_assignment is not None:
                    q_flat = v_layer[fixed_assignment]
                elif lsq_enabled:
                    # Per-channel needs the original shape so that output
                    # channels stay the leading dimension for broadcasting.
                    q_source = param.data if lsq_per_channel else w_flat
                    q_flat, _ = quantize_weight(
                        q_source,
                        lsq_scales[q_state].detach(),
                        lsq_integer_codebooks[q_state],
                    )
                    q_flat = q_flat.reshape(-1)
                else:
                    _, q_flat = _quantize_with_deadzone(
                        w_flat,
                        v_layer,
                        apply_pruning_deadzone=(prune_mode == "deadzone"),
                    )

                # Optimization-driven pruning: apply the cached z-based mask so the
                # loss sees the quantized + pruned model.  One epoch of dual warmup
                # (epoch >= entropy_warmup_epochs + 1) lets xi converge before we
                # start pruning the forward.
                if (prune_mode == "z"
                        and epoch >= entropy_warmup_epochs
                        and z_prune_masks[q_state] is not None):
                    q_flat = torch.where(
                        z_prune_masks[q_state],
                        torch.zeros_like(q_flat),
                        q_flat,
                    )

                # Perspective (test_113): magnitude pruning, coherent between the
                # fake-quant forward and evaluation.  Starts after one epoch of
                # grid-reset warmup so the L1 push has begun shrinking weights.
                if (prune_mode == "magnitude"
                        and epoch >= entropy_warmup_epochs + 1):
                    prune_mask = _mask_to_apply(
                        q_state, w_flat, v_list[q_state]
                    )
                    q_flat = torch.where(
                        prune_mask,
                        torch.zeros_like(q_flat),
                        q_flat,
                    )

                param.data.copy_(q_flat.view_as(param))

        return backups

    def _restore_fake_quantized_weights_(backups):
        """
        Restores floating-point weights after backward.
        Gradients remain those computed through the quantized forward.
        """
        with torch.no_grad():
            for param, original in backups:
                param.data.copy_(original)            

    def _accumulate_lsq_task_scale_grad_(backups):
        """Build LSQ scale gradients and apply the LSQ weight STE clipping."""
        if not lsq_enabled:
            return
        with torch.no_grad():
            for q_state, (param, original) in enumerate(backups):
                if param.grad is None:
                    continue
                # Per-channel reduces the scale gradient channel by channel, so
                # it needs the original shape; per-tensor keeps the flat call.
                if lsq_per_channel:
                    grad_weight = original
                    grad_upstream = param.grad.detach()
                else:
                    grad_weight = original.reshape(-1)
                    grad_upstream = param.grad.detach().reshape(-1)
                scale_gradient, in_range = task_scale_gradient(
                    grad_weight,
                    grad_upstream,
                    lsq_scales[q_state],
                    lsq_integer_codebooks[q_state],
                    normalize=lsq_grad_scaling,
                )
                if lsq_scales[q_state].grad is None:
                    lsq_scales[q_state].grad = scale_gradient.reshape_as(
                        lsq_scales[q_state]
                    )
                else:
                    lsq_scales[q_state].grad.add_(scale_gradient)
                lsq_task_scale_grad_last[q_state] = float(
                    scale_gradient.mean().item()
                )
                param.grad.mul_(in_range.reshape_as(param.grad))

    def _reestimate_batchnorm_stats_():
        """Re-estimate BN running statistics with the deployed quantized weights.

        The reference protocol evaluates after averaging 50 training-batch
        statistics with BN momentum 1. We aggregate those statistics across DDP
        ranks and then install the same buffers on every rank.
        """
        if bn_recalibration_batches <= 0:
            return 0
        bn_modules = [
            module
            for module in model.modules()
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
            and module.track_running_stats
        ]
        if not bn_modules:
            return 0

        was_training = model.training
        original_momenta = [module.momentum for module in bn_modules]
        mean_sums = [torch.zeros_like(module.running_mean) for module in bn_modules]
        var_sums = [torch.zeros_like(module.running_var) for module in bn_modules]
        fake_quant_backups = _fake_quantize_weights_for_forward_()
        model.eval()
        for module in bn_modules:
            module.momentum = 1.0
            module.training = True

        batches = 0
        autocast_ctx = torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=(device.type == "cuda"),
        )
        with torch.no_grad(), autocast_ctx:
            for inputs, _ in trainloader:
                if batches >= bn_recalibration_batches:
                    break
                inputs = inputs.to(device, non_blocking=True)
                if device.type == "cuda":
                    inputs = inputs.contiguous(memory_format=torch.channels_last)
                model(inputs)
                for idx, module in enumerate(bn_modules):
                    mean_sums[idx].add_(module.running_mean)
                    var_sums[idx].add_(module.running_var)
                batches += 1

        count = torch.tensor(float(batches), device=device)
        if dist.is_initialized():
            dist.all_reduce(count, op=dist.ReduceOp.SUM)
            for mean_sum, var_sum in zip(mean_sums, var_sums):
                dist.all_reduce(mean_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(var_sum, op=dist.ReduceOp.SUM)
        denominator = count.clamp_min(1.0)
        for module, momentum, mean_sum, var_sum in zip(
            bn_modules, original_momenta, mean_sums, var_sums
        ):
            module.running_mean.copy_(mean_sum / denominator)
            module.running_var.copy_(var_sum / denominator)
            module.momentum = momentum

        _restore_fake_quantized_weights_(fake_quant_backups)
        if was_training:
            model.train()
        else:
            model.eval()
        return batches

    # Diagnostic norm of the currently accumulated gradients on this rank.
    def _grad_norm_from_current_grads() -> float:
        with torch.no_grad():
            total_sq = torch.zeros((), device=device)
            for p_ in model.parameters():
                if p_.grad is not None:
                    total_sq += p_.grad.detach().float().pow(2).sum()
            return torch.sqrt(total_sq).item()

    # DDP correctness check: all ranks should have identical parameters after
    # each optimizer step.  A non-zero range indicates a rank-local update.
    def _param_checksum_range() -> float | None:
        if not dist.is_initialized():
            return None
        with torch.no_grad():
            checksum = torch.zeros((), device=device)
            for p_ in model.parameters():
                checksum += p_.detach().float().sum()
            gathered = [torch.zeros_like(checksum) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered, checksum)
            values = torch.stack(gathered)
            return (values.max() - values.min()).item()
        
    def _percentiles_large_tensor(x: torch.Tensor, probs: list[float]) -> torch.Tensor:
        x = x.flatten().float()
        n = x.numel()
        vals = []
        for p in probs:
            k = max(1, min(n, int(round(p * (n - 1))) + 1))
            vals.append(torch.kthvalue(x, k).values)
        return torch.stack(vals)    

    def _pack_quant_indices(idx: torch.Tensor, C_local: int) -> bytes:
        """
        Pack quantization indices.

        For C <= 16, stores two 4-bit indices per byte.
        For larger C, falls back to uint8/uint16/int32.
        """
        if idx.numel() == 0:
            return b""

        if C_local <= 16:
            arr = idx.detach().to(torch.uint8).cpu().numpy()

            if arr.size % 2 == 1:
                arr = np.pad(arr, (0, 1), constant_values=0)

            packed = ((arr[0::2] & 0x0F) << 4) | (arr[1::2] & 0x0F)
            return packed.astype(np.uint8).tobytes()

        if C_local <= 256:
            return idx.detach().to(torch.uint8).cpu().numpy().tobytes()

        if C_local <= 65536:
            return idx.detach().to(torch.uint16).cpu().numpy().tobytes()

        return idx.detach().to(torch.int32).cpu().numpy().tobytes()

    def _index_bits_for_C(C_local: int) -> int:
        return int(math.ceil(math.log2(C_local)))      

    def _make_quant_levels_without_zero(lo: torch.Tensor, hi: torch.Tensor, C_local: int, device):
        """
        Builds a quantization grid with C_local levels between lo and hi, but with an explicit zero dead-zone.  
        The grid is symmetric around zero, but the positive and negative halves are separate to allow an explicit gap around zero.  
        The gap size is proportional to the layer range (hi - lo) to avoid degenerate grids for small layers.
        """
        gap_ratio = 0.05

        lo = lo.to(device)
        hi = hi.to(device)

        r_layer = 0.5 * (hi - lo).abs()
        gap = gap_ratio * r_layer

        lo = torch.minimum(lo, -gap)
        hi = torch.maximum(hi, gap)

        n_neg = C_local // 2
        n_pos = C_local - n_neg

        neg = torch.linspace(lo.item(), -gap.item(), steps=n_neg, device=device)
        pos = torch.linspace(gap.item(), hi.item(), steps=n_pos, device=device)

        levels = torch.cat([neg, pos]).to(dtype=torch.float32)
        return levels      

    def _quantize_with_deadzone(
        w_flat: torch.Tensor,
        v_layer: torch.Tensor,
        apply_pruning_deadzone: bool,
    ):
        """
        Quantizes weights to the nearest non-zero level in v_layer.
        If apply_pruning_deadzone=True, weights near zero are quantized to exactly zero, creating a dead-zone that induces sparsity.
        """
        levels = v_layer
        C_local = levels.numel()
        boundaries = (levels[:-1] + levels[1:]) / 2

        q_idx = torch.bucketize(w_flat, boundaries, right=False).clamp_(0, C_local - 1)
        q_flat = levels[q_idx]

        if apply_pruning_deadzone:
            if delta < 0:
                alpha_delta = (-delta) / (1.0 - delta)
            else:
                alpha_delta = 0.0

            pruning_threshold = gamma * alpha_delta * levels.abs().min()

            prune_mask = w_flat.abs() <= pruning_threshold
            q_flat = torch.where(prune_mask, torch.zeros_like(q_flat), q_flat)

        return q_idx, q_flat
    
    def _dual_zero_mask_from_knapsack(
        w_flat: torch.Tensor,
        v_layer: torch.Tensor,
        xi_layer: torch.Tensor,
    ):
        """
        Builds a pruning mask from the sparse-aware FISTA/knapsack solution.

        The zero mass is

            z_i = 1 - sum_b x_{i,b}

        Since z_i is a relaxed quantity, we convert it to a hard pruning mask by
        rounding its total mass:

            num_pruned = round(sum_i z_i)

        and pruning the num_pruned weights with largest z_i.

        This is used only for evaluation/compression diagnostics.
        """
        xi_layer = xi_layer.to(dtype=torch.float32, device=device)
        C_local = v_layer.numel()

        if xi_layer.numel() != C_local + 1:
            raise ValueError(
                f"dual-zero pruning requires xi_layer with length C+1={C_local+1}, "
                f"got {xi_layer.numel()}."
            )

        if device.type == "cuda":
            x_star, _, _ = knapsack_specialized_pruning_sparse_leonardo(
                xi_layer,
                v_layer,
                w_flat,
                C_local,
                device,
                delta,
            )
        else:
            x_star, _, _ = knapsack_specialized_pruning(
                xi_layer,
                v_layer,
                w_flat,
                C_local,
                device,
                delta,
            )

        if x_star.dim() == 2 and x_star.size(1) == 3:
            idx_left = x_star[:, 0].to(dtype=torch.long, device=device)
            idx_right = x_star[:, 1].to(dtype=torch.long, device=device)
            theta = x_star[:, 2].to(dtype=torch.float32, device=device)

            sum_x = theta.clone()
            mask_diff = idx_right != idx_left

            if mask_diff.any():
                sum_x[mask_diff] += 1.0 - theta[mask_diff]
        else:
            sum_x = x_star.sum(dim=1)

        z_mass = (1.0 - sum_x).clamp_(0.0, 1.0)

        num_weights = z_mass.numel()

        # Use the old gamma/deadzone rule only to determine HOW MANY weights to prune.
        # Then rank candidates using both dual zero mass and magnitude.
        if delta < 0:
            alpha_delta = (-delta) / (1.0 - delta)
        else:
            alpha_delta = 0.0

        pruning_threshold = gamma * alpha_delta * v_layer.abs().min()
        gamma_budget_mask = w_flat.abs() <= pruning_threshold

        num_pruned = int(gamma_budget_mask.sum().item())
        num_pruned = max(0, min(num_weights, num_pruned))

        abs_w = w_flat.abs()
        threshold_safe = pruning_threshold.clamp_min(1e-12)

        normalized_abs = abs_w / threshold_safe

        score = z_mass / (
            (normalized_abs + dual_zero_score_eps) ** dual_zero_abs_power
        )

        # Do not allow dual-zero ranking to prune weights that are too large.
        candidate_threshold = dual_zero_candidate_multiplier * threshold_safe
        candidate_mask = abs_w <= candidate_threshold

        score = torch.where(
            candidate_mask,
            score,
            torch.full_like(score, float("-inf")),
        )

        prune_mask = torch.zeros_like(z_mass, dtype=torch.bool)

        if num_pruned > 0:
            if num_pruned >= num_weights:
                prune_mask.fill_(True)
            else:
                topk_idx = torch.topk(
                    score,
                    k=num_pruned,
                    largest=True,
                    sorted=False,
                ).indices
                prune_mask[topk_idx] = True

        return prune_mask, z_mass

    def _z_prune_mask(w_flat, v_layer, xi_layer, step_scale=None):
        """
        Optimization-driven pruning mask.

        Solves the per-weight sparse-aware knapsack for the current xi and prunes
        the weights whose relaxed solution assigns most of the mass to the zero
        symbol:

            z_i = 1 - sum_b x_{i,b} > z_prune_threshold

        delta enters the knapsack costs (xi_b - xi_zero - delta), so both HOW MANY
        and WHICH weights are pruned come from the optimization, not a magnitude
        threshold.
        """
        # Under the perspective path, z must come from the actual METaQ
        # per-weight mass y*(w), not from the legacy knapsack dead-zone
        # controlled by ``delta``.  This is the path used by the T1/T2
        # ablations: z_i = 1-y_i and the hard deployment rule is z_i>0.5.
        if use_perspective:
            y_star = _perspective_y_star(w_flat, v_layer, step_scale=step_scale)
            return (1.0 - y_star).clamp_(0.0, 1.0) > z_prune_threshold

        xi_layer = xi_layer.to(dtype=torch.float32, device=device)
        C_local = v_layer.numel()

        if device.type == "cuda":
            x_star, _, _ = knapsack_specialized_pruning_sparse_leonardo(
                xi_layer, v_layer, w_flat, C_local, device, delta,
            )
        else:
            x_star, _, _ = knapsack_specialized_pruning(
                xi_layer, v_layer, w_flat, C_local, device, delta,
            )

        if x_star.dim() == 2 and x_star.size(1) == 3:
            idx_left = x_star[:, 0].to(dtype=torch.long, device=device)
            idx_right = x_star[:, 1].to(dtype=torch.long, device=device)
            theta = x_star[:, 2].to(dtype=torch.float32, device=device)
            sum_x = theta.clone()
            mask_diff = idx_right != idx_left
            if mask_diff.any():
                sum_x[mask_diff] += 1.0 - theta[mask_diff]
        else:
            sum_x = x_star.sum(dim=1)

        z_mass = (1.0 - sum_x).clamp_(0.0, 1.0)
        return z_mass > z_prune_threshold

    def _quantize_with_dual_zero_pruning(
        w_flat: torch.Tensor,
        v_layer: torch.Tensor,
        xi_layer: torch.Tensor,
    ):
        """
        Quantizes to the nearest non-zero bucket, but decides pruning using
        the dual-zero mass z_i from the sparse-aware knapsack solution.
        """
        q_idx, q_flat = _quantize_with_deadzone(
            w_flat,
            v_layer,
            apply_pruning_deadzone=False,
        )

        if prune_mode == "z":
            prune_mask = _z_prune_mask(w_flat, v_layer, xi_layer)
            z_mass = None
        else:
            prune_mask, z_mass = _dual_zero_mask_from_knapsack(
                w_flat,
                v_layer,
                xi_layer,
            )

        q_flat = torch.where(prune_mask, torch.zeros_like(q_flat), q_flat)

        return q_idx, q_flat, z_mass

    # ------------------------------------------------------------------
    # Perspective reformulation (test_113), entropy_coeff == 0 closed form.
    #
    # Per-weight subproblem   min_x sum_b xi_b x_b + perspective_coeff w^2/y + sparsity_coeff y  reduces,
    # when the entropy costs vanish (entropy_coeff = 0 => xi = 0), to the closed form
    # verified in CheckCorrectnessPerspectiveAlgorithm.ipynb (TEST E):
    #
    #     y*(w) = clamp( |w| * sqrt(perspective_coeff/sparsity_coeff), [ymin, 1] )
    #     ymin  = |w| / (max positive bucket if w>=0 else |min negative bucket|)
    #     dphi/dw = 2 * perspective_coeff * w / y*      (perspective ridge + L1 sparsity push)
    #
    # Near w = 0 the push tends to 2*sqrt(perspective_coeff*sparsity_coeff)*sign(w): an L1 term that drives
    # small weights to 0, so the magnitude pruning below removes them cleanly.
    # ------------------------------------------------------------------
    def _perspective_y_star(w_flat, v_layer, step_scale=None):
        aw = w_flat.abs()
        # Feasibility floor: w/y must be representable, i.e. w/y in [min v, max v].
        # For w>0 that means y >= w / max_positive_bucket; for w<0, y >= |w| / |min
        # bucket|.  Using max_b|v_b| for both signs would allow an infeasible target
        # on the side whose extreme bucket is smaller in magnitude.
        pos_max = v_layer.max().clamp_min(1e-12)      # most positive bucket (>0)
        neg_absmax = v_layer.min().abs().clamp_min(1e-12)  # |most negative bucket| (>0)
        side_max = torch.where(w_flat >= 0, pos_max, neg_absmax)
        if step_scale is not None:
            # v_layer holds integer codes under per-channel quantization, so the
            # real extreme level of weight i is step_scale_i * q_edge.
            side_max = (side_max * step_scale).clamp_min(1e-12)
        ymin_c = (aw / side_max).clamp_(max=1.0)
        if sparsity_coeff > 0.0:
            scale = math.sqrt(perspective_coeff / sparsity_coeff)
            y_int = (aw * scale).clamp_(max=1.0)
        else:
            # No sparsity term: keep full mass (plain ridge, y* = 1).
            y_int = torch.ones_like(aw)
        # y* = max(interior stationary point, feasibility floor), capped at 1.
        return torch.maximum(y_int, ymin_c)

    def _perspective_ridge_grad(w_flat, v_layer):
        y_star = _perspective_y_star(w_flat, v_layer)
        g = torch.zeros_like(w_flat)
        nz = y_star > 0.0
        g[nz] = 2.0 * perspective_coeff * w_flat[nz] / y_star[nz]
        return g

    def _zero_entropy_perspective_gradients(w_flat, v_layer, scale=None):
        """Exact dphi/dw and equality multiplier when the bucket costs are zero.

        With entropy_coeff=0 the inner problem reduces to a scalar minimization in y, so
        invoking the general hull/knapsack solver for every weight is unnecessary.
        The equality multiplier is zero away from the representability floor and
        follows the boundary KKT condition on that floor.

        ``scale`` carries one step size per weight under per-channel
        quantization. ``v_layer`` is then the INTEGER codebook, so the real
        extreme level of weight i is scale_i * q_edge.
        """
        y_star = _perspective_y_star(w_flat, v_layer, step_scale=scale)
        beta_constraint = torch.zeros_like(w_flat)
        beta_star = torch.zeros_like(w_flat)
        nonzero = w_flat != 0
        if not nonzero.any():
            return beta_star, beta_constraint

        positive_edge = v_layer[-1]
        negative_edge = v_layer[0]
        edge_v = torch.where(w_flat >= 0, positive_edge, negative_edge)
        if scale is not None:
            edge_v = edge_v * scale        # real extreme level scale_i*q_edge
        representation_floor = (
            w_flat.abs() / edge_v.abs().clamp_min(1e-12)
        ).clamp(max=1.0)
        at_floor = nonzero & torch.isclose(
            y_star,
            representation_floor,
            rtol=1e-5,
            atol=1e-7,
        )
        boundary_multiplier = sparsity_coeff / edge_v - perspective_coeff * edge_v
        beta_constraint[at_floor] = boundary_multiplier[at_floor]
        beta_star[nonzero] = (
            beta_constraint[nonzero]
            + 2.0 * perspective_coeff * w_flat[nonzero] / y_star[nonzero]
        )
        return beta_star, beta_constraint

    def _perspective_mag_threshold(w_flat, v_layer, ts=None):
        # The magnitude threshold below which a weight is pruned.
        #  - target_sparsity > 0: per-layer, prune the smallest ts fraction of
        #    |w| (kthvalue gives that exact quantile).  ts defaults to
        #    target_sparsity but can be a smaller ramped value (test_118).  This
        #    gives DIRECT, per-layer control of sparsity (conv and FC alike),
        #    instead of the uniform ratio*min|v| threshold.
        #  - otherwise: the fixed threshold mag_prune_ratio * min_b|v_b|.
        explicit_target = ts is not None
        if ts is None:
            ts = target_sparsity
        if explicit_target and ts <= 0.0:
            # An explicit zero comes from a schedule/controller and means no
            # magnitude pruning. Keep exact zeros pruned if another operator
            # created them, but do not fall back to mag_prune_ratio.
            return torch.zeros((), dtype=w_flat.dtype, device=w_flat.device)
        if ts and ts > 0.0:
            aw = w_flat.abs()
            n = aw.numel()
            k_total = int(ts * n)
            k_total = max(1, min(n - 1, k_total))
            # test_136: the quantile must be taken over the NON-ZERO weights.
            # test_135 introduced the proximal operator, which sends weights to
            # EXACTLY zero; the plain kthvalue over all |w| then breaks down as
            # soon as the zero mass exceeds ts, because the ts-quantile IS zero,
            # the threshold collapses to 0, and (the threshold being frozen for
            # the epoch) the loss steps afterwards nudge those weights off exact
            # zero so that "|w| <= 0" prunes almost nothing.  That is how test_135
            # reported a deployed sparsity of 2.86% against a 20% target while the
            # per-layer conv fractions sat at 0.20.
            # Correct semantics: the zeros already count towards the target, and
            # any remainder is taken from the smallest non-zero weights.
            n_zero = int((aw <= 0).sum().item())
            if k_total <= n_zero:
                # the zero mass alone already meets (or exceeds) the target:
                # threshold 0 prunes exactly those weights
                return torch.zeros((), dtype=aw.dtype, device=aw.device)
            aw_nz = aw[aw > 0]
            k = max(1, min(aw_nz.numel(), k_total - n_zero))
            return torch.kthvalue(aw_nz, k).values
        return mag_prune_ratio * v_layer.abs().min()

    def _perspective_prune_mask(w_flat, v_layer, thr=None):
        # Magnitude pruning (Frangioni): prune weights whose |w| is below the
        # threshold.  thr may be a frozen per-epoch value (cheaper); if None it
        # is computed on the spot.
        if thr is None:
            thr = _perspective_mag_threshold(w_flat, v_layer)
        return w_flat.abs() <= thr

    def _refresh_prune_thresholds():
        """test_136: recompute the frozen thresholds from the CURRENT weights.

        Only needed in prox mode.  The proximal operator keeps sending weights to
        exactly zero throughout the epoch, so the zero mass at evaluation time is
        much larger than it was when the threshold was frozen at epoch start; the
        stale threshold then implies a deployed sparsity that has nothing to do
        with the target (test_135 swung between 2.86% and 43.24% against a 20%
        target).  Recomputing right before the metrics costs one kthvalue per
        layer, once per epoch.
        """
        with torch.no_grad():
            for p_idx, param in enumerate(params_for_quant):
                ts_layer = effective_ts_by_layer[p_idx]
                if ts_layer is None:
                    continue
                target_prune_thresholds[p_idx] = _perspective_mag_threshold(
                    param.detach().reshape(-1), v_list[p_idx], ts_layer
                )

    def _mask_to_apply(p_idx, w_flat, v_layer):
        # test_144: use the frozen index set if we have one; otherwise the usual
        # |w| <= threshold recomputed from current weights.
        if freeze_mask and frozen_prune_masks[p_idx] is not None:
            return frozen_prune_masks[p_idx]
        return _perspective_prune_mask(
            w_flat, v_layer, target_prune_thresholds[p_idx]
        )

    def _mask_in_parameter_shape(mask, param):
        """Return a logical-shape mask without relying on parameter contiguity.

        AlexNet convolution weights use channels-last storage. Flattening such a
        tensor with ``reshape(-1)`` can allocate a copy, so writing through the
        flattened result does not necessarily modify the original parameter.
        Reshaping the contiguous boolean mask to the parameter's logical shape
        and applying it directly is correct for either memory format.
        """
        if mask.numel() != param.numel():
            raise ValueError(
                f"Pruning mask has {mask.numel()} entries for a parameter with "
                f"{param.numel()} entries."
            )
        return mask.reshape(param.shape)

    def _train_sparse_mask_diagnostics():
        """Return (masked entries, nonzero violations, maximum absolute value)."""
        total_masked = 0
        nonzero_violations = 0
        max_abs = 0.0
        with torch.no_grad():
            for p_idx, param in enumerate(params_for_quant):
                if effective_ts_by_layer[p_idx] is None:
                    continue
                w_flat = param.detach().reshape(-1)
                mask = _mask_to_apply(p_idx, w_flat, v_list[p_idx])
                shaped_mask = _mask_in_parameter_shape(mask, param)
                masked_values = param.detach()[shaped_mask]
                total_masked += masked_values.numel()
                if masked_values.numel() > 0:
                    nonzero_violations += int(
                        torch.count_nonzero(masked_values).item()
                    )
                    max_abs = max(max_abs, masked_values.abs().max().item())
        return total_masked, nonzero_violations, max_abs

    def _quantize_with_magnitude_pruning(w_flat, v_layer, thr=None, frozen_mask=None):
        q_idx, q_flat = _quantize_with_deadzone(
            w_flat, v_layer, apply_pruning_deadzone=False,
        )
        prune_mask = (frozen_mask if frozen_mask is not None
                      else _perspective_prune_mask(w_flat, v_layer, thr))
        q_flat = torch.where(prune_mask, torch.zeros_like(q_flat), q_flat)
        return q_idx, q_flat

    # test_143: parse the iterative prune-and-heal schedule, if given.
    # stages: list of (reach_epoch, hold_until_epoch, target_vec), 1-based epochs.
    parsed_stages = None
    if sparsity_schedule:
        parsed_stages = []
        for chunk in sparsity_schedule.split(";"):
            reach_s, hold_s, vec_s = chunk.strip().split(":")
            vec = [float(x) for x in vec_s.split(",")]
            parsed_stages.append((int(reach_s), int(hold_s), vec))
        if local_rank == 0:
            print(f"[SPARSITY SCHEDULE] {len(parsed_stages)} stage: "
                  + " | ".join(f"reach{r}->hold{h} fc6={v[5]:.2f}"
                               for r, h, v in parsed_stages), flush=True)

    def _staged_targets(epoch_1based):
        """Per-layer effective sparsity at this epoch, from the staged schedule.

        Within each stage the target ramps from the previous stage's vector (or 0
        for the first) up to `reach`, then holds flat through `hold`.  The ramp
        uses the SAME concave shape (frac ** sparsity_ramp_power) as the smooth
        ramp of the stable runs (test_121/124): this is what makes test_143 a
        clean control -- it reproduces the stable ramp to each level and only
        ADDS the plateaus.  For the small inter-stage jumps (span 1) the shape is
        irrelevant.
        """
        n_layers = len(parsed_stages[0][2])
        prev_vec = [0.0] * n_layers
        stage_start = entropy_warmup_epochs + 1        # 1-based first prune epoch
        for reach, hold, vec in parsed_stages:
            if epoch_1based <= hold:
                span = max(1, reach - (stage_start - 1))
                frac = min(1.0, max(0.0, (epoch_1based - (stage_start - 1)) / span))
                frac = frac ** sparsity_ramp_power
                return [pv + frac * (tv - pv) for pv, tv in zip(prev_vec, vec)]
            prev_vec = vec
            stage_start = hold + 1
        return list(parsed_stages[-1][2])              # past the last stage: hold

    def _activate_trainable_codebook_():
        """Freeze current assignments, project weights, and reset their momentum."""
        nonlocal codebook_active, v_list

        codebook_infos = []
        kmeans_infos = []
        with torch.no_grad():
            for p_idx, param in enumerate(params_for_quant):
                w_flat = param.detach().reshape(-1)
                if centroid_kmeans_iterations > 0:
                    if (not dist.is_initialized()) or local_rank == 0:
                        (
                            learned_centroids,
                            _,
                            iterations_run,
                            initial_mse,
                            final_mse,
                            final_shift,
                        ) = _lloyd_codebook_1d(
                            w_flat,
                            v_list[p_idx],
                            centroid_kmeans_iterations,
                        )
                    else:
                        learned_centroids = torch.empty_like(v_list[p_idx])
                        iterations_run = 0
                        initial_mse = 0.0
                        final_mse = 0.0
                        final_shift = 0.0
                    if dist.is_initialized():
                        dist.broadcast(learned_centroids, src=0)
                    v_list[p_idx] = learned_centroids
                    if local_rank == 0:
                        kmeans_infos.append(
                            (
                                p_idx,
                                iterations_run,
                                initial_mse,
                                final_mse,
                                final_shift,
                            )
                        )

                assignment, projected = _quantize_with_deadzone(
                    w_flat,
                    v_list[p_idx],
                    apply_pruning_deadzone=False,
                )
                fixed_codebook_assignments[p_idx] = assignment.detach().clone()
                param.data.copy_(projected.view_as(param))

                # Dynamic QAT has populated a distinct momentum buffer for every
                # shadow weight. Keeping it would immediately untie the codebook.
                optimizer.state.pop(param, None)

                counts = torch.bincount(
                    assignment,
                    minlength=C_by_layer[p_idx],
                )
                nonempty = counts[counts > 0]
                codebook_infos.append(
                    (
                        p_idx,
                        int(nonempty.numel()),
                        int(nonempty.min().item()) if nonempty.numel() else 0,
                        int(nonempty.max().item()) if nonempty.numel() else 0,
                    )
                )

        codebook_active = True
        if local_rank == 0:
            print(
                f"[TRAINABLE CENTROIDS FREEZE] epoch={epoch + 1}, "
                f"dynamic_qat_epochs={max(0, epoch - entropy_warmup_epochs)}",
                flush=True,
            )
            if centroid_kmeans_iterations > 0:
                print(
                    "[CENTROID KMEANS] "
                    "entries=(layer,iterations,initial_mse,final_mse,"
                    f"final_max_shift), values={kmeans_infos}",
                    flush=True,
                )
            print(
                "[TRAINABLE CENTROIDS INIT] "
                "entries=(layer,nonempty,min_count,max_count), "
                f"values={codebook_infos}",
                flush=True,
            )

    # test_133: cumulative entropy "exposure" = sum over epochs of entropy_coeff_current*lr
    # (lr in units of 1e-4).  This is the accumulated displacement the entropy term
    # imposes on the weights, and across tests 128/129/130 the accuracy collapse
    # lined up with it at ~1.6.  Logging it per epoch makes the wall directly
    # readable instead of reconstructed by hand from the entropy_coeff/lr traces.
    exposure_cum = 0.0

    if evaluate_initial_model:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        _dist_barrier()
        pretrained_accuracy = test_accuracyGPU(model, testloader, device)
        if local_rank == 0:
            print(
                f"[PRETRAINED BASELINE] fp_accuracy={pretrained_accuracy:.4f}",
                flush=True,
            )
        _dist_barrier()

    for epoch in range(n_epochs):
        should_eval_epoch = ((epoch + 1) % metrics_interval == 0) or (epoch == n_epochs - 1)

        # entropy_coeff schedule: no entropy during warmup, then a gentle exponential ramp
        # from entropy_coeff/8 to entropy_coeff.  This avoids abruptly injecting a large custom
        # gradient after many epochs of standard training.
        if entropy_coeff > 0 and epoch >= entropy_warmup_epochs:
            if flat_schedule:
                # test_133: no ramp.  entropy_coeff is on at full value from the first
                # post-warmup epoch, so every active epoch spends exactly the same
                # amount of exposure (entropy_coeff*lr) and the rate is flat by construction.
                entropy_coeff_current = entropy_coeff
            else:
                t = epoch - entropy_warmup_epochs
                entropy_coeff_current = entropy_coeff * (1.0 - np.exp(-t)) + (entropy_coeff / 8.0) * np.exp(-t)
        else:
            entropy_coeff_current = 0.0

        if (
            train_centroids
            and centroid_freeze_epoch > 0
            and epoch + 1 >= centroid_freeze_epoch
        ):
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            if local_rank == 0 and epoch + 1 == centroid_freeze_epoch:
                print(
                    f"[CENTROID PHASE LR RESET] epoch={epoch + 1}, base_lr={lr}",
                    flush=True,
                )

        # Exposure spent during THIS epoch, using the LR the epoch actually runs
        # with (the scheduler steps at the end of the epoch, so read it here).
        epoch_lr = optimizer.param_groups[0]['lr']
        exposure_epoch = entropy_coeff_current * (epoch_lr / 1e-4)
        exposure_cum += exposure_epoch

        # Compression phase: adapt the per-layer quantization grid ONCE
        # (test_111).  A FIXED grid keeps the dual's xi converged and the z
        # pruning mask stable; re-adapting every epoch (test_108/110) made xi
        # stale at each epoch start, the forward mask shifted, and the net
        # collapsed.
        codebook_initialized_now = False
        # LSQ already owns a valid learned codebook, so expose it to METaQ from
        # epoch 1 independently of the entropy_coeff warm-up. The warm-up controls only the
        # entropy term; delaying this initialization also delayed perspective_coeff/sparsity_coeff and made
        # test_165's first epoch incomparable with the entropy_coeff=0 baseline. Fixed grids
        # retain their historical warm-up behaviour.
        if (lsq_enabled or epoch >= entropy_warmup_epochs) and not grid_reset_done:
            with torch.no_grad():
                if lsq_enabled:
                    for scale in lsq_scales:
                        if dist.is_initialized():
                            dist.broadcast(scale, src=0)
                    _refresh_lsq_metaq_grids_()
                    grid_reset_done = True
                    if local_rank == 0:
                        print(
                            f"[LSQ GRID INIT] epoch={epoch + 1}, "
                            f"scales={[float(s.detach().mean().item()) for s in lsq_scales]}, "
                            f"deployment_levels={C_by_layer}, "
                            f"metaq_nonzero_buckets={metaq_C_by_layer}",
                            flush=True,
                        )
                else:
                    new_v_list = []
                    grid_infos = []
                    for p_idx, param in enumerate(params_for_quant):
                        w_layer = param.detach().reshape(-1).float()
                        lo, hi = _percentiles_large_tensor(w_layer, [0.001, 0.999])
                        if (hi - lo).abs() < 1e-12:
                            center = 0.5 * (lo + hi)
                            lo = center - 1e-6
                            hi = center + 1e-6
                        v_layer = _make_quant_levels_without_zero(
                            lo, hi, C_by_layer[p_idx], device
                        )
                        if dist.is_initialized():
                            dist.broadcast(v_layer, src=0)
                        new_v_list.append(v_layer)
                        grid_infos.append((p_idx, w_layer.numel(), lo.item(), hi.item()))
                    v_list = new_v_list
                    grid_reset_done = True
                    if local_rank == 0:
                        print(
                            f"[GRID READAPT] epoch={epoch + 1}, "
                            f"num_tensors={len(v_list)}, first_tensors={grid_infos[:4]}",
                            flush=True
                        )

        should_activate_codebook = (
            train_centroids
            and grid_reset_done
            and not codebook_active
            and (
                centroid_freeze_epoch == 0
                or epoch + 1 >= centroid_freeze_epoch
            )
        )
        if should_activate_codebook:
            _activate_trainable_codebook_()
            codebook_initialized_now = True

        if codebook_initialized_now:
            _dist_barrier()
            with torch.no_grad():
                codebook_initial_accuracy = test_accuracyGPU(
                    model,
                    testloader,
                    device,
                )
            _dist_barrier()
            if local_rank == 0:
                print(
                    "[TRAINABLE CENTROIDS BASELINE] "
                    f"accuracy_before_updates={codebook_initial_accuracy}",
                    flush=True,
                )

        # test_111: freeze the optimization-driven pruning mask ONCE per epoch
        # (from the converged xi) and hold it for every forward of this epoch.
        # Stable counterpart of the per-step recompute that destabilised test_110.
        # Pruning starts after one epoch of dual warmup.
        if prune_mode == "z" and grid_reset_done and epoch >= entropy_warmup_epochs:
            with torch.no_grad():
                total_n_fz = 0
                total_pruned_fz = 0
                for p_idx, param in enumerate(params_for_quant):
                    w_layer = param.detach().reshape(-1)
                    mask = _z_prune_mask(
                        w_layer, v_list[p_idx], xi_list[p_idx],
                        _metaq_scale_flat(p_idx, param),
                    )
                    z_prune_masks[p_idx] = mask
                    total_n_fz += mask.numel()
                    total_pruned_fz += int(mask.sum().item())
                if local_rank == 0:
                    frac_fz = 100.0 * total_pruned_fz / max(1, total_n_fz)
                    print(
                        f"[FROZEN Z-MASK] epoch={epoch + 1}, "
                        f"pruned={frac_fz:.2f}%, delta={delta}",
                        flush=True
                    )

        # test_116: freeze the per-layer magnitude threshold for target_sparsity
        # mode once per epoch (kthvalue is expensive to recompute every step).
        # The frozen threshold is used by both the fake-quant forward and eval.
        #
        # test_118: linear sparsity ramp.  Instead of pruning the full
        # target_sparsity in one shot (which the net cannot recover from at high
        # sparsity), the effective target grows linearly from 0 to target_sparsity
        # over sparsity_warmup_epochs, then holds.  Deep-Compression-style gradual
        # pruning.  sparsity_warmup_epochs <= 0 reproduces the one-shot behavior.
        #
        # test_122: non-uniform per-layer sparsity.  When conv_sparsity and
        # fc_sparsity are both set, conv weights (4D) and FC weights (2D) get
        # different per-layer targets: conv layers (few params, accuracy-sensitive)
        # are pruned gently, FC layers (~96% of AlexNet params, redundant) hard.
        # The global sparsity stays high (FC-dominated) while conv is protected.
        # test_124: full per-layer sparsity list (Deep-Compression style) takes
        # priority over the conv/fc grouping, which takes priority over the uniform
        # target_sparsity.
        per_layer_sparsity = (conv_sparsity is not None and fc_sparsity is not None)
        sparsity_active = (
            (target_sparsity and target_sparsity > 0.0)
            or per_layer_sparsity
            or (layer_sparsity is not None)
            or (parsed_stages is not None)
        )
        if (use_perspective and sparsity_active
                and grid_reset_done and epoch >= entropy_warmup_epochs + 1):
            first_prune_epoch = entropy_warmup_epochs + 1
            if sparsity_warmup_epochs and sparsity_warmup_epochs > 0:
                steps_into = epoch - first_prune_epoch + 1
                frac = min(1.0, max(0.0, steps_into / float(sparsity_warmup_epochs)))
                # test_121: ramp profile.  power = 1.0 -> linear; power < 1 (e.g.
                # 0.5) -> concave, i.e. small sparsity increments near the target,
                # giving the net more adaptation time in the hard 60->85% region.
                ramp_frac = frac ** sparsity_ramp_power
            else:
                ramp_frac = 1.0
            # Staged and adiabatic schedules override the smooth ramp.
            staged_vec = _staged_targets(epoch + 1) if parsed_stages is not None else None
            adiabatic_vec = (
                [target * adiabatic_progress for target in adiabatic_base_targets]
                if adiabatic_enabled
                else None
            )
            with torch.no_grad():
                for p_idx, param in enumerate(params_for_quant):
                    if adiabatic_vec is not None:
                        effective_ts = adiabatic_vec[p_idx]
                    elif staged_vec is not None:
                        effective_ts = staged_vec[p_idx]
                    else:
                        if layer_sparsity is not None:
                            layer_ts = layer_sparsity[p_idx]
                        elif per_layer_sparsity:
                            layer_ts = conv_sparsity if param.dim() == 4 else fc_sparsity
                        else:
                            layer_ts = target_sparsity
                        effective_ts = layer_ts * ramp_frac
                    w_layer = param.detach().reshape(-1)
                    target_prune_thresholds[p_idx] = _perspective_mag_threshold(
                        w_layer, v_list[p_idx], effective_ts
                    )
                    # test_136: remember the target this threshold was built for,
                    # so it can be refreshed at evaluation time in prox mode.
                    effective_ts_by_layer[p_idx] = effective_ts
                    # test_144: (re)freeze the boolean mask only when the target
                    # changed -- i.e. during the sub-ramp and at each stage jump.
                    # While a plateau holds, the target is constant and the mask
                    # is left fixed, so the net heals into a stable sparse set.
                    if freeze_mask:
                        prev_ts = frozen_mask_ts[p_idx]
                        if prev_ts is None or abs(effective_ts - prev_ts) > 1e-6:
                            frozen_prune_masks[p_idx] = _perspective_prune_mask(
                                w_layer, v_list[p_idx],
                                target_prune_thresholds[p_idx],
                            ).clone()
                            frozen_mask_ts[p_idx] = effective_ts
            if local_rank == 0:
                if adiabatic_vec is not None:
                    tgt_str = (
                        f"adiabatic_progress={adiabatic_progress:.4f}, effective=["
                        + ",".join(f"{v:.4f}" for v in adiabatic_vec)
                        + "]"
                    )
                elif staged_vec is not None:
                    tgt_str = "staged=[" + ",".join(f"{v:.2f}" for v in staged_vec) + "]"
                elif layer_sparsity is not None:
                    tgt_str = f"per_layer={list(layer_sparsity)}*{ramp_frac:.3f}"
                elif per_layer_sparsity:
                    tgt_str = f"conv={conv_sparsity}*{ramp_frac:.3f}, fc={fc_sparsity}*{ramp_frac:.3f}"
                else:
                    tgt_str = f"effective_target_sparsity={target_sparsity * ramp_frac:.4f}"
                print(
                    f"[SPARSITY RAMP] epoch={epoch + 1}, {tgt_str}, "
                    f"ramp_frac={ramp_frac:.4f}, ramp_power={sparsity_ramp_power}",
                    flush=True
                )

        for param_group in optimizer.param_groups:
            param_group["weight_decay"] = wd_init

        if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        start_time_global = time.time()
        train_batches = 0
        entropy_steps = 0
        last_loss_grad_norm = None
        last_custom_beta_norm = None
        last_entropy_fraction = None
        last_centroid_grad_norm = None
        last_centroid_grad_abs_max = None
        lsq_task_scale_grad_last = [0.0] * num_param_tensors
        lsq_metaq_scale_grad_last = [0.0] * num_param_tensors
        centroid_displacement_max = 0.0
        centroid_displacement_mean_sum = 0.0
        centroid_displacement_count = 0
        # test_126 diagnostics for the perspective entropy path (Causa B).  We
        # accumulate, per epoch, the norm of the APPLIED entropy update vs the
        # loss-gradient norm (is the entropy signal strong or weak?), and the
        # common-mode fraction of beta* = |mean(beta)|*sqrt(N)/||beta|| in [0,1]
        # (1 => a pure uniform shift that erodes accuracy without reshaping the
        # weight histogram, hence flat H_Q).  Also track the xi dual range.
        persp_entropy_norm_sum = 0.0
        persp_grad_norm_sum = 0.0
        persp_commonmode_sum = 0.0
        persp_diag_count = 0
        persp_xi_min = None
        persp_xi_max = None
        persp_xi_mean_sum = 0.0
        persp_xi_pinned_sum = 0.0
        # test_135: proximal-step diagnostics.  prox_displacement is the mean
        # |z* - u| per application, i.e. how far the operator actually moves the
        # weights; prox_zero_frac is the fraction it sends exactly to zero (the
        # prox prunes natively, via the shifted soft-threshold).  Both are needed
        # to calibrate gamma, which is a brand-new knob with no prior.
        prox_disp_sum = 0.0
        prox_zero_sum = 0.0
        prox_diag_count = 0
        # test_112 diagnostic: recompute the z>0.5 fraction via _z_prune_mask
        # (the SAME path used at eval) right after each dual step, to compare it
        # against FISTA's internal frac_sum_x_lt_0_5 and pin the 68%-vs-4% gap.
        z_recompute_sum = torch.zeros((), device=device)
        z_recompute_count = 0

        for i, data in enumerate(trainloader, 0):
            if steps_per_epoch is not None and i >= steps_per_epoch:
                break

            global_step += 1
            train_batches += 1

            inputs, targets = data
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if device.type == "cuda":
                inputs = inputs.contiguous(memory_format=torch.channels_last)

            optimizer.zero_grad(set_to_none=True)
            if scale_optimizer is not None:
                scale_optimizer.zero_grad(set_to_none=True)

            # Fake-quantized forward:
            # after the per-layer grids have been initialized, the loss sees
            # the quantized model, while the optimizer still updates FP32 weights.
            fake_quant_backups = _fake_quantize_weights_for_forward_()

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=(device.type == "cuda")
            ):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                if teacher_model is not None:
                    # Soft knowledge distillation. The teacher runs on the same
                    # inputs with its own full-precision weights; the student is
                    # asked to reproduce the whole class distribution, not just
                    # the label, which is far more information and is precisely
                    # the function we want the quantized model to keep.
                    with torch.no_grad():
                        teacher_logits = teacher_model(inputs)
                    student_log_probs = torch.nn.functional.log_softmax(
                        outputs.float() / distill_tau, dim=1
                    )
                    teacher_probs = torch.nn.functional.softmax(
                        teacher_logits.float() / distill_tau, dim=1
                    )
                    # The tau^2 factor keeps the gradient magnitude of this term
                    # independent of the temperature.
                    distill_loss = torch.nn.functional.kl_div(
                        student_log_probs,
                        teacher_probs,
                        reduction="batchmean",
                    ) * (distill_tau ** 2)
                    task_loss_last = float(loss.detach())
                    distill_loss_last = float(distill_loss.detach())
                    loss = (
                        (1.0 - distill_alpha) * loss
                        + distill_alpha * distill_loss
                    )

            loss.backward()

            _restore_fake_quantized_weights_(fake_quant_backups)
            _accumulate_lsq_task_scale_grad_(fake_quant_backups)

            is_last_configured_step = steps_per_epoch is not None and (i + 1) >= steps_per_epoch
            capture_last_norms = local_rank == 0 and should_eval_epoch and is_last_configured_step
            if capture_last_norms:
                last_loss_grad_norm = _grad_norm_from_current_grads()

            # Perspective ridge/sparsity gradient (test_113). For legacy fixed-grid
            # runs retain the closed-form weight gradient. In joint LSQ-METaQ runs
            # solve the xi=0 inner problem explicitly: its constraint multiplier is
            # also needed for the exact envelope gradient with respect to the LSQ
            # scale, including the representability boundary.
            # perspective_coeff/sparsity_coeff are active as soon as the codebook exists. In particular,
            # entropy_warmup_epochs postpones entropy_coeff only; it must not silently
            # disable the other METaQ terms during the warm-up (test_166).
            if (use_perspective and entropy_coeff_current == 0
                    and (perspective_coeff != 0.0 or sparsity_coeff != 0.0)
                    and grid_reset_done):
                with torch.no_grad():
                    for p_idx, param in enumerate(params_for_quant):
                        if param.grad is None:
                            continue
                        w_layer = param.detach().reshape(-1)
                        if joint_lsq_metaq:
                            # Under per-channel `scale` is one step size per
                            # weight, so every expression below stays elementwise
                            # and torch.clamp's scalar bounds become tensors.
                            scale = (
                                _metaq_scale_flat(p_idx, param)
                                if lsq_per_channel
                                else lsq_scales[p_idx].detach().clamp_min(1e-12)
                            )
                            q_full = lsq_integer_codebooks[p_idx]
                            normalized = w_layer / scale
                            qn = q_full[0]
                            qp = q_full[-1]
                            metaq_weight = torch.minimum(
                                torch.maximum(w_layer, qn * scale), qp * scale
                            )
                            in_range = (normalized >= qn) & (normalized <= qp)
                            clipping_scale_direction = torch.where(
                                normalized < qn,
                                qn,
                                torch.where(
                                    normalized > qp,
                                    qp,
                                    torch.zeros_like(normalized),
                                ),
                            )
                            beta_star, beta_constraint = (
                                _zero_entropy_perspective_gradients(
                                    metaq_weight,
                                    v_list[p_idx],
                                    scale=(
                                        scale if lsq_per_channel else None
                                    ),
                                )
                            )
                            param.grad.add_((beta_star * in_range).view_as(param))
                            scale_phi_grad = metaq_scale_gradient(
                                beta_constraint,
                                metaq_weight,
                                lsq_scales[p_idx],
                            )
                            scale_phi_grad = _add_clipping_scale_term(
                                scale_phi_grad,
                                beta_star,
                                clipping_scale_direction,
                                lsq_scales[p_idx],
                            )
                            if lsq_scales[p_idx].grad is None:
                                lsq_scales[p_idx].grad = scale_phi_grad.reshape_as(
                                    lsq_scales[p_idx]
                                )
                            else:
                                lsq_scales[p_idx].grad.add_(scale_phi_grad)
                            lsq_metaq_scale_grad_last[p_idx] = float(
                                scale_phi_grad.mean().item()
                            )
                        else:
                            ridge_g = _perspective_ridge_grad(
                                w_layer, v_list[p_idx]
                            )
                            param.grad.add_(ridge_g.view_as(param))

            # test_125: perspective entropy (entropy_coeff > 0).  Run the general perspective
            # FISTA every entropy_every steps: it updates the bucket duals xi and
            # returns beta* = dphi/dw (ridge + entropy) per weight, which we apply
            # directly.  NOT YET GPU-VALIDATED (the inner solver is offline-verified;
            # the xi-dynamics are new) -- treat the first run as a smoke test.
            if (use_perspective and not use_prox and entropy_coeff_current > 0
                    and grid_reset_done and epoch >= entropy_warmup_epochs
                    and (global_step % entropy_every == 0)):
                with torch.no_grad():
                    for p_idx, param in enumerate(params_for_quant):
                        if param.grad is None:
                            continue
                        w_layer = param.detach().reshape(-1).to(device)
                        C_layer = metaq_C_by_layer[p_idx]
                        metaq_scale = _metaq_scale_flat(p_idx, param)
                        if lsq_enabled:
                            scale = (
                                metaq_scale
                                if lsq_per_channel
                                else lsq_scales[p_idx].detach().clamp_min(1e-12)
                            )
                            q_full = lsq_integer_codebooks[p_idx]
                            normalized = w_layer / scale
                            qn = q_full[0]
                            qp = q_full[-1]
                            metaq_weight = torch.minimum(
                                torch.maximum(w_layer, qn * scale), qp * scale
                            )
                            metaq_weight_in_range = (
                                (normalized >= qn) & (normalized <= qp)
                            )
                            clipping_scale_direction = torch.where(
                                normalized < qn,
                                qn,
                                torch.where(
                                    normalized > qp,
                                    qp,
                                    torch.zeros_like(normalized),
                                ),
                            )
                        else:
                            metaq_weight = w_layer
                            metaq_weight_in_range = torch.ones_like(
                                w_layer, dtype=torch.bool
                            )
                            clipping_scale_direction = None

                        (
                            xi_list[p_idx],
                            beta_star,
                            beta_constraint,
                        ) = FISTA_perspective_leonardo(
                            xi_list[p_idx], v_list[p_idx], metaq_weight, C_layer,
                            float(metaq_weight.numel()), lower_c,
                            perspective_coeff, entropy_coeff_current, sparsity_coeff,
                            subgradient_step, device, max_iterations,
                            dual_step,
                            scale=metaq_scale,
                        )
                        if joint_lsq_metaq:
                            # Exact envelope contribution for
                            # F=L_QAT+phi(clip_LSQ(w,s),s). Outside the LSQ
                            # range, dclip/dw=0 and dclip/ds=q_edge.
                            beta_for_weight = beta_star * metaq_weight_in_range
                            param.grad.add_(beta_for_weight.view_as(param))
                            scale_phi_grad = metaq_scale_gradient(
                                beta_constraint,
                                metaq_weight,
                                lsq_scales[p_idx],
                            )
                            scale_phi_grad = _add_clipping_scale_term(
                                scale_phi_grad,
                                beta_star,
                                clipping_scale_direction,
                                lsq_scales[p_idx],
                            )
                            if lsq_scales[p_idx].grad is None:
                                lsq_scales[p_idx].grad = scale_phi_grad.reshape_as(
                                    lsq_scales[p_idx]
                                )
                            else:
                                lsq_scales[p_idx].grad.add_(scale_phi_grad)
                            lsq_metaq_scale_grad_last[p_idx] = float(
                                scale_phi_grad.mean().item()
                            )
                        # winsorize beta* to clip stray coordinates, THEN auto-scale
                        # to the gradient norm exactly as the (proven-stable) old dual
                        # path does.  Winsorizing to the median only caps outliers; it
                        # cannot stop a near-uniform bulk shift of beta*, which blew the
                        # weights up in the first test_125 (p50 -> -808 the moment entropy_coeff
                        # turned on).  Rescaling ties the perspective update to
                        # entropy_coeff_current * ref_grad_norm, bounding its magnitude.
                        bstar = (
                            beta_star * metaq_weight_in_range
                            if joint_lsq_metaq
                            else beta_star
                        ).float().reshape(-1)
                        # test_126 diagnostic: common-mode fraction of the RAW beta*
                        # (intrinsic ~0.70), scale-invariant.  Logged to document why
                        # we center below.
                        n_b = bstar.numel()
                        raw_norm = bstar.norm().clamp_min(1e-12)
                        cm_ratio = (bstar.mean().abs() * math.sqrt(n_b) / raw_norm).item()
                        # test_127: center beta* (remove the all-ones component).  Its
                        # intrinsic common-mode is a rigid translation of the weight
                        # cloud: it wrecks accuracy without reshaping the bucket
                        # histogram (H_Q flat), and on GPU it feeds back (uniform shift
                        # -> skewed counts -> xi ran away to +-450, test_126).  Keeping
                        # only the differential part preserves the clustering signal.
                        grad_layer_norm = param.grad.detach().float().norm()
                        if not joint_lsq_metaq:
                            bstar = bstar - bstar.mean()
                            bscale = bstar.abs().median().clamp_min(1e-12)
                            bstar = bstar.clamp(min=-beta_clip_k * bscale, max=beta_clip_k * bscale)
                            if entropy_ref_grad_norm[p_idx] is None:
                                entropy_ref_grad_norm[p_idx] = grad_layer_norm.detach().clone()
                            ref_grad_norm = entropy_ref_grad_norm[p_idx]
                            bstar_norm = bstar.norm().clamp_min(1e-12)
                            bstar = (entropy_coeff_current * ref_grad_norm / bstar_norm) * bstar
                            param.grad.add_(bstar.view_as(param))
                        if local_rank == 0:
                            persp_entropy_norm_sum += bstar.norm().item()
                            persp_grad_norm_sum += grad_layer_norm.item()
                            persp_commonmode_sum += cm_ratio
                            persp_diag_count += 1
                            xi_b_now = (xi_list[p_idx][1:]
                                        if xi_list[p_idx].numel() == C_layer + 1
                                        else xi_list[p_idx])
                            lo, hi = xi_b_now.min().item(), xi_b_now.max().item()
                            persp_xi_min = lo if persp_xi_min is None else min(persp_xi_min, lo)
                            persp_xi_max = hi if persp_xi_max is None else max(persp_xi_max, hi)
                            persp_xi_mean_sum += xi_b_now.mean().item()
                            # test_134: fraction of buckets sitting exactly on a
                            # clamp bound.  This is the direct convergence check:
                            # in test_133 it was ~15/16 every step (bang-bang
                            # chattering, dual never converged).  Near 0 means the
                            # dual found an interior solution.
                            _ln2 = math.log(2.0)
                            entropy_coeff_safe = max(float(entropy_coeff_current), 1e-12)
                            _xlo = entropy_coeff_safe * (math.log(max(lower_c, 1e-12)) + 1.0) / _ln2
                            _xhi = entropy_coeff_safe * (math.log(max(float(w_layer.numel()), 1e-12)) + 1.0) / _ln2
                            # test_168: use a tolerance relative to the actual
                            # dual interval. The previous absolute floor of 1e-6
                            # covered a large part of the whole interval when
                            # entropy_coeff <= 1e-7, falsely reporting interior xi values as
                            # pinned (73% in test_167).
                            _xi_scale = max(
                                abs(_xlo), abs(_xhi),
                                abs(_xhi - _xlo), 1e-12,
                            )
                            _tol = 1e-6 * _xi_scale
                            persp_xi_pinned_sum += (
                                ((xi_b_now - _xlo).abs() < _tol)
                                | ((xi_b_now - _xhi).abs() < _tol)
                            ).float().mean().item()
                    entropy_steps += 1
                    if dist.is_initialized():
                        # Re-sync grads: beta* is computed from each rank's local xi,
                        # which was updated via non-deterministic scatter_add BEFORE the
                        # xi broadcast below, so beta* differs slightly across ranks.
                        # Without this all_reduce the ranks' params drift apart
                        # (ddp_param_checksum_range grew 0 -> 1.4 over epochs 2-5).
                        # Mirrors the old dual path.
                        for param in model.parameters():
                            if param.grad is not None:
                                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                                param.grad.div_(dist.get_world_size())
                        for xi_layer in xi_list:
                            dist.broadcast(xi_layer, src=0)

            beta_tensor = None
            # Entropy/FISTA is the expensive part of the method.  Applying it
            # every N global steps keeps the computational cost controlled.
            run_dual = (
                grid_reset_done
                and (global_step % entropy_every == 0)
                and (entropy_coeff_current > 0 or prune_mode == "z")
                # The entropy dual (old FISTA/knapsack with xi_zero/delta) is not
                # part of the perspective path; the general entropy_coeff>0 perspective solver
                # is a later step.  Never run the old dual under use_perspective.
                and not use_perspective
            )
            if run_dual:
                # The dual (FISTA on xi) runs to keep the pruning mass z meaningful
                # even at entropy_coeff == 0 (pure optimization-driven pruning, no entropy
                # gradient).  The entropy gradient is added only when entropy_coeff > 0.
                with torch.no_grad():
                    # FISTA is applied independently to each parameter tensor.
                    # Each tensor has its own xi, v, and possibly C.
                    zeta *= 1 + l
                    l = l / 1.5

                    beta_norm_sq = torch.zeros((), device=device)
                    grad_norm_sq = torch.zeros((), device=device)

                    for p_idx, param in enumerate(params_for_quant):
                        if param.grad is None:
                            continue

                        w_layer = param.detach().reshape(-1).to(device)
                        C_layer = C_by_layer[p_idx]

                        # upper_c should be local because c_star represents
                        # bucket occupancies for this tensor, not for the whole network.
                        upper_c_layer = float(w_layer.numel())

                        if entropy_optimizer == 'FISTA':
                            xi_list[p_idx], beta_layer = FISTA_leonardo(
                                xi_list[p_idx],
                                v_list[p_idx],
                                w_layer,
                                C_layer,
                                upper_c_layer,
                                lower_c,
                                delta,
                                subgradient_step,
                                device,
                                max_iterations,
                                pruning
                            )
                        elif entropy_optimizer == 'PROXIMAL BM':
                            xi_list[p_idx], beta_layer = ProximalBM(
                                xi_list[p_idx],
                                v_list[p_idx],
                                w_layer,
                                C_layer,
                                upper_c_layer,
                                lower_c,
                                delta,
                                zeta,
                                subgradient_step,
                                device,
                                max_iterations,
                                pruning
                            )
                        else:
                            raise ValueError(f"Unsupported entropy optimizer: {entropy_optimizer}")

                        # The entropy GRADIENT is applied only when entropy_coeff > 0.  When
                        # entropy_coeff == 0 the dual still ran above (xi updated) so z stays
                        # meaningful, but nothing is added to the weights: the
                        # pruning is then purely optimization-driven.
                        if entropy_coeff_current > 0:
                            # Sanitize (winsorize) + auto-scale the entropy subgradient.
                            beta_dir = (-beta_layer).float().reshape(-1)
                            beta_scale = beta_dir.abs().median().clamp_min(1e-12)
                            beta_dir = beta_dir.clamp(
                                min=-beta_clip_k * beta_scale,
                                max=beta_clip_k * beta_scale,
                            )
                            grad_layer_norm = param.grad.detach().float().norm()
                            if entropy_ref_grad_norm[p_idx] is None:
                                entropy_ref_grad_norm[p_idx] = grad_layer_norm.detach().clone()
                            ref_grad_norm = entropy_ref_grad_norm[p_idx]
                            beta_dir_norm = beta_dir.norm().clamp_min(1e-12)
                            entropy_update = (
                                entropy_coeff_current * ref_grad_norm / beta_dir_norm
                            ) * beta_dir
                            param.grad.add_(entropy_update.view_as(param))
                            beta_norm_sq += entropy_update.pow(2).sum()
                            grad_norm_sq += grad_layer_norm.pow(2)

                        # test_112 diagnostic: same path as eval, same xi, same
                        # moment as FISTA's internal z debug.
                        if prune_mode == "z":
                            z_recompute_sum = z_recompute_sum + _z_prune_mask(
                                w_layer, v_list[p_idx], xi_list[p_idx]
                            ).float().mean()
                            z_recompute_count += 1

                    if dist.is_initialized():
                        # The entropy term is added after loss.backward()'s DDP sync,
                        # so re-sync the grads only when we actually added entropy.
                        if entropy_coeff_current > 0:
                            for param in model.parameters():
                                if param.grad is not None:
                                    dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                                    param.grad.div_(dist.get_world_size())

                        for xi_layer in xi_list:
                            dist.broadcast(xi_layer, src=0)

                    entropy_steps += 1

                    if local_rank == 0 and entropy_coeff_current > 0:
                        last_custom_beta_norm = torch.sqrt(beta_norm_sq).item()
                        last_entropy_fraction = torch.sqrt(
                            beta_norm_sq / grad_norm_sq.clamp_min(1e-12)
                        ).item()

            # test_146: train the sparse SUBNETWORK directly (Deep-Compression
            # style).  Until now every run has fake-quantized+pruned in the
            # FORWARD but restored the dense weights before the step, so the
            # pruned weights kept a non-zero gradient and drifted back -- we were
            # optimizing the DENSE net and deploying a masked version of it.  Here
            # we instead hold the pruned positions at exactly zero and give them NO
            # gradient, so the optimizer moves ONLY the survivors: the loss is
            # minimized over the sparse subnetwork we actually deploy.  Done right
            # before the step, and the pruned weights are re-zeroed right after, so
            # they never drift.  The mask is the same one used in the forward.
            if train_sparse and grid_reset_done and epoch >= entropy_warmup_epochs + 1:
                with torch.no_grad():
                    for p_idx, param in enumerate(params_for_quant):
                        if param.grad is None:
                            continue
                        w_flat = param.detach().reshape(-1)
                        mask = _mask_to_apply(p_idx, w_flat, v_list[p_idx])
                        param.grad.masked_fill_(
                            _mask_in_parameter_shape(mask, param.grad),
                            0.0,
                        )

            # Deep Compression trained quantization: cluster membership is fixed,
            # and the derivative of a shared centroid is the SUM of the
            # derivatives of all weights assigned to it. DDP has already averaged
            # every per-weight loss gradient across ranks; the bucket reduction
            # therefore preserves the correct global gradient. GPU scatter_add is
            # nondeterministic, so rank 0 broadcasts the tiny centroid-gradient
            # vectors before they are expanded back to the tied weights.
            if train_centroids and codebook_active:
                with torch.no_grad():
                    # The fixed-codebook phase optimizes only shared centroids.
                    # Biases and other unquantized parameters are skipped entirely
                    # (grad=None also prevents stale momentum from moving them).
                    for param in all_params:
                        if id(param) not in quant_param_ids:
                            param.grad = None
                    centroid_grad_sq = torch.zeros((), device=device)
                    centroid_grad_abs_max = torch.zeros((), device=device)
                    for p_idx, param in enumerate(params_for_quant):
                        if param.grad is None:
                            continue
                        assignment = fixed_codebook_assignments[p_idx]
                        centroid_grad = _aggregate_codebook_gradient(
                            param.grad.detach().reshape(-1),
                            assignment,
                            C_by_layer[p_idx],
                        )
                        if dist.is_initialized():
                            dist.broadcast(centroid_grad, src=0)
                        if capture_last_norms:
                            centroid_grad_sq += centroid_grad.float().pow(2).sum()
                            centroid_grad_abs_max = torch.maximum(
                                centroid_grad_abs_max,
                                centroid_grad.abs().max(),
                            )
                        centroid_grad.mul_(centroid_lr_scale)
                        param.grad.copy_(
                            centroid_grad[assignment].view_as(param)
                        )
                    if capture_last_norms:
                        last_centroid_grad_norm = torch.sqrt(centroid_grad_sq).item()
                        last_centroid_grad_abs_max = centroid_grad_abs_max.item()

            optimizer.step()

            if scale_optimizer is not None:
                if dist.is_initialized():
                    for scale in lsq_scales:
                        if scale.grad is not None:
                            dist.all_reduce(scale.grad, op=dist.ReduceOp.SUM)
                            scale.grad.div_(dist.get_world_size())
                scale_optimizer.step()
                with torch.no_grad():
                    for scale in lsq_scales:
                        scale.clamp_(min=1e-12)
                        if dist.is_initialized():
                            dist.broadcast(scale, src=0)
                    _refresh_lsq_metaq_grids_()

            if train_centroids and codebook_active:
                with torch.no_grad():
                    for p_idx, param in enumerate(params_for_quant):
                        old_assignment = fixed_codebook_assignments[p_idx]
                        old_centroids = v_list[p_idx]
                        centroids, assignment, projected, relabel = _rebuild_sorted_codebook(
                            param.detach().reshape(-1),
                            old_assignment,
                            old_centroids,
                        )
                        if dist.is_initialized():
                            # The mean reduction above uses GPU atomics. Broadcast
                            # both the compact codebook and its old->new relabelling
                            # from rank 0 so all ranks remain bit-identical.
                            dist.broadcast(centroids, src=0)
                            dist.broadcast(relabel, src=0)
                            assignment = relabel[old_assignment]
                            projected = centroids[assignment]
                        centroid_displacement = (
                            centroids[relabel] - old_centroids
                        ).abs()
                        centroid_displacement_max = max(
                            centroid_displacement_max,
                            centroid_displacement.max().item(),
                        )
                        centroid_displacement_mean_sum += (
                            centroid_displacement.mean().item()
                        )
                        centroid_displacement_count += 1
                        v_list[p_idx] = centroids
                        fixed_codebook_assignments[p_idx] = assignment
                        param.copy_(projected.view_as(param))

            if train_sparse and grid_reset_done and epoch >= entropy_warmup_epochs + 1:
                with torch.no_grad():
                    for p_idx, param in enumerate(params_for_quant):
                        w_flat = param.detach().reshape(-1)
                        mask = _mask_to_apply(p_idx, w_flat, v_list[p_idx])
                        param.data.masked_fill_(
                            _mask_in_parameter_shape(mask, param),
                            0.0,
                        )

            # test_135: PROXIMAL step.  The optimizer above has just taken a plain
            # gradient step on the LOSS ALONE (in prox mode the beta*-into-grad
            # branch is skipped); phi is now applied as its own operator directly
            # on the weights:
            #     w <- prox_{gamma*phi}(w - lr*grad L)
            # This is proximal gradient descent on the SAME objective L + phi --
            # the model does not change, only the algorithm.  It removes the two
            # properties that test_134 pinned as the bottleneck: the entropy
            # displacement no longer competes with the loss gradient inside one
            # summed direction, and it is no longer throttled by the learning rate
            # (its size is set by gamma).  All the beta* hygiene of tests 125-127
            # (centering, winsorizing, rescaling to entropy_coeff*ref_grad_norm) is
            # unnecessary here: a prox is a well-defined operator, not a term to
            # be dosed against the loss.
            if (use_prox and entropy_coeff_current > 0
                    and grid_reset_done and epoch >= entropy_warmup_epochs
                    and epoch >= prox_start_epoch
                    and (global_step % entropy_every == 0)):
                with torch.no_grad():
                    for p_idx, param in enumerate(params_for_quant):
                        u_layer = param.detach().reshape(-1).to(device)
                        C_layer = C_by_layer[p_idx]
                        xi_list[p_idx], _, _ = FISTA_prox_leonardo(
                            xi_list[p_idx], v_list[p_idx], u_layer, C_layer,
                            float(u_layer.numel()), lower_c,
                            perspective_coeff, entropy_coeff_current, sparsity_coeff,
                            prox_gamma, device, max_iterations, dual_step,
                        )
                        # xi is updated through a non-deterministic scatter_add, so
                        # it can differ across ranks.  param.data is identical on
                        # every rank (DDP synced the grads and every rank ran the
                        # same optimizer step), so broadcasting xi and only THEN
                        # computing the applied z* keeps the weights bit-identical
                        # across ranks -- the same drift that had to be fixed in
                        # test_125, arriving here by a different route.
                        if dist.is_initialized():
                            dist.broadcast(xi_list[p_idx], src=0)
                        xi_b = (xi_list[p_idx][1:]
                                if xi_list[p_idx].numel() == C_layer + 1
                                else xi_list[p_idx])
                        _, z_star, y_star = prox_perspective_leonardo(
                            xi_b, v_list[p_idx], u_layer, C_layer, device,
                            perspective_coeff, sparsity_coeff, prox_gamma,
                        )
                        if local_rank == 0:
                            prox_disp_sum += (z_star - u_layer).abs().mean().item()
                            prox_zero_sum += (z_star.abs() <= 1e-12).float().mean().item()
                            prox_diag_count += 1
                        param.copy_(z_star.view_as(param))

        if not (
            train_centroids
            and centroid_freeze_epoch > 0
            and codebook_active
        ):
            scheduler.step()
            if scale_scheduler is not None:
                scale_scheduler.step()

        training_time_without_metrics = round(time.time() - start_time_global)

        if should_eval_epoch:
            # test_136: in prox mode refresh the pruning thresholds against the
            # weights as they are NOW, before any metric is computed from them.
            if use_prox:
                _refresh_prune_thresholds()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            _dist_barrier()

            bn_batches_used = _reestimate_batchnorm_stats_()
            if local_rank == 0 and bn_batches_used > 0:
                print(
                    f"[BN RECALIBRATION] local_batches={bn_batches_used}, "
                    f"world_size={dist.get_world_size() if dist.is_initialized() else 1}",
                    flush=True,
                )

            with torch.no_grad():
                accuracy = test_accuracyGPU(model, testloader, device)
            if local_rank == 0:
                accuracies.append(accuracy)

            _dist_barrier()

            with torch.no_grad():
                # Backup the full model.
                # Quantize only weight tensors; keep bias tensors in FP32.
                w_backup_parts = []
                flat_q_parts = []
                flat_s_parts = []

                q_idx_layers = []
                q_vals_layers = []

                s_idx_layers = []
                s_vals_layers = []

                quant_state = 0

                for full_idx, p_ in enumerate(all_params):
                    w_layer = p_.detach().reshape(-1).clone()
                    w_backup_parts.append(w_layer)

                    if full_idx in quant_param_indices and QuantizationType == "center":
                        v_layer = v_list[quant_state]
                        quant_levels = _current_quant_levels(quant_state)

                        if not use_quantization:
                            # Pruning-only control: the dense and surviving
                            # values stay FP32. The sparse stream differs only by
                            # the same magnitude mask used during training.
                            q_idx_layer = None
                            s_idx_layer = None
                            q_layer = w_layer.clone()
                            if (
                                use_perspective
                                and effective_ts_by_layer[quant_state] is not None
                            ):
                                prune_mask = _mask_to_apply(
                                    quant_state,
                                    w_layer,
                                    v_layer,
                                )
                                s_layer = torch.where(
                                    prune_mask,
                                    torch.zeros_like(w_layer),
                                    w_layer,
                                )
                            else:
                                s_layer = w_layer.clone()
                        else:
                            # Pure quantization: no pruning deadzone.
                            fixed_assignment = fixed_codebook_assignments[quant_state]
                            if train_centroids and fixed_assignment is not None:
                                q_idx_layer = fixed_assignment
                                q_layer = quant_levels[q_idx_layer]
                            elif lsq_enabled:
                                # w_layer is flat here, so under per-channel the
                                # step sizes are expanded to one per weight.
                                eval_scale = lsq_scales[quant_state].detach()
                                if lsq_per_channel:
                                    eval_scale = expand_scale_flat(
                                        eval_scale.clamp_min(1e-12),
                                        params_for_quant[quant_state],
                                    )
                                q_layer, integer_assignment = quantize_weight(
                                    w_layer,
                                    eval_scale,
                                    lsq_integer_codebooks[quant_state],
                                )
                                q_idx_layer = (
                                    integer_assignment
                                    - lsq_integer_codebooks[quant_state][0]
                                ).to(torch.long)
                            else:
                                q_idx_layer, q_layer = _quantize_with_deadzone(
                                    w_layer,
                                    quant_levels,
                                    apply_pruning_deadzone=False,
                                )

                            # Sparse quantization: pruning decided by the
                            # configured sparse-aware rule.
                            if train_centroids and fixed_assignment is not None:
                                s_idx_layer = q_idx_layer
                                s_layer = q_layer.clone()
                            elif use_perspective:
                                if prune_mode == "z":
                                    # Use exactly the mask used by the fake-
                                    # quantized training forward. Recomputing a
                                    # different magnitude mask here was the
                                    # source of the 99.5% vs 17% discrepancy.
                                    prune_mask = z_prune_masks[quant_state]
                                    if prune_mask is None:
                                        prune_mask = _z_prune_mask(
                                            w_layer,
                                            v_layer,
                                            xi_list[quant_state],
                                            _metaq_scale_flat(
                                                quant_state,
                                                params_for_quant[quant_state],
                                            ),
                                        )
                                else:
                                    prune_mask = (
                                        frozen_prune_masks[quant_state]
                                        if freeze_mask
                                        and frozen_prune_masks[quant_state] is not None
                                        else _perspective_prune_mask(
                                            w_layer,
                                            v_layer,
                                            target_prune_thresholds[quant_state],
                                        )
                                    )
                                s_idx_layer = q_idx_layer
                                s_layer = torch.where(
                                    prune_mask,
                                    torch.zeros_like(q_layer),
                                    q_layer,
                                )
                            else:
                                s_idx_layer, s_layer, _ = _quantize_with_dual_zero_pruning(
                                    w_layer,
                                    v_layer,
                                    xi_list[quant_state],
                                )

                            s_layer[s_layer.abs() <= sparsity_threshold] = 0.0

                        q_idx_layers.append(q_idx_layer)
                        q_vals_layers.append(q_layer)

                        s_idx_layers.append(s_idx_layer)
                        s_vals_layers.append(s_layer)

                        quant_state += 1

                    else:
                        # Biases and non-quantized tensors remain floating point.
                        q_layer = w_layer.clone()
                        s_layer = w_layer.clone()

                    flat_q_parts.append(q_layer)
                    flat_s_parts.append(s_layer)

                w_backup = torch.cat(w_backup_parts)
                flat_q = torch.cat(flat_q_parts)
                flat_s = torch.cat(flat_s_parts)

                # Distinguish the requested logical pruning mask from the
                # zeros that happen to arise after LSQ quantization.
                z_mask_total = 0
                z_mask_pruned = 0
                if prune_mode == "z":
                    for mask in z_prune_masks:
                        if mask is not None:
                            z_mask_total += mask.numel()
                            z_mask_pruned += int(mask.sum().item())

            if local_rank == 0:
                with torch.no_grad():
                    entropy_total = 0.0
                    q_bytes_chunks = []
                    nz_idx_bytes_chunks = []
                    nz_masks = []

                    total_n = 0
                    total_nz = 0

                    # Distribution of non-zero quantized indices in the sparse representation.
                    # This is used to estimate H(bucket | nonzero).
                    nz_index_counts = torch.zeros(max_C, dtype=torch.float64, device=device)
                    sparse_symbol_counts = torch.zeros(max_C + 1, dtype=torch.float64, device=device)

                    for q_idx_layer, q_vals_layer, s_idx_layer, s_vals_layer, C_layer in zip(
                        q_idx_layers,
                        q_vals_layers,
                        s_idx_layers,
                        s_vals_layers,
                        C_by_layer,
                    ):
                        if use_quantization:
                            # Dense quantized stream: bucket indices.
                            counts = torch.bincount(
                                q_idx_layer, minlength=C_layer
                            ).to(torch.float32)
                            probs = counts / counts.sum()
                            probs = probs[probs > 0]
                            entropy_total += float(
                                (-(probs * torch.log2(probs)).sum() * counts.sum()).item()
                            )
                            q_bytes_chunks.append(
                                _pack_quant_indices(q_idx_layer, C_layer)
                            )
                            nz_mask_layer = s_vals_layer.abs() > sparsity_threshold
                        else:
                            # FP32 pruning-only stream. Values are accounted for
                            # at 32 bits each below; do not materialize hundreds
                            # of MB of bytes merely to describe the baseline.
                            nz_mask_layer = s_vals_layer != 0.0
                        nz_masks.append(nz_mask_layer)

                        total_n += s_vals_layer.numel()
                        total_nz += int(nz_mask_layer.sum().item())

                        if use_quantization:
                            nz_idx_layer = s_idx_layer[nz_mask_layer]
                            num_zero_layer = int((~nz_mask_layer).sum().item())
                            sparse_symbol_counts[0] += num_zero_layer

                            if nz_idx_layer.numel() > 0:
                                counts_nz = torch.bincount(
                                    nz_idx_layer,
                                    minlength=C_layer,
                                ).to(dtype=torch.float64)
                                sparse_symbol_counts[1:1 + C_layer] += counts_nz
                                nz_index_counts[:C_layer] += counts_nz

                            nz_idx_bytes_chunks.append(
                                _pack_quant_indices(nz_idx_layer, C_layer)
                            )
                        else:
                            # Raw nonzero FP32 storage is counted from total_nz.
                            pass

                    quantized_entropy = (
                        round(entropy_total) + 1
                        if use_quantization
                        else total_n * 32
                    )
                    entropies.append(quantized_entropy)

                    # Metadata for fixed linear grids is two fp32 endpoints per
                    # layer. A trained codebook must instead store every learned
                    # fp32 centroid; include that cost in all reported ratios.
                    if not use_quantization:
                        metadata_bits = 0
                    elif train_centroids and codebook_active:
                        metadata_bits = 32 + 32 + sum(C_by_layer) * 32
                    elif lsq_enabled:
                        # One fp32 step size per tensor; signed integer ranges
                        # are determined globally by C and need no per-layer data.
                        # Per-channel quantization ships one step size per output
                        # channel instead, and those bits must be paid for here
                        # or the reported ratio would be dishonest.
                        num_step_sizes = (
                            sum(int(s.numel()) for s in lsq_scales)
                            if (lsq_enabled and lsq_per_channel)
                            else num_param_tensors
                        )
                        metadata_bits = 32 + 32 + num_step_sizes * 32
                    else:
                        metadata_bits = 32 + 32 + num_param_tensors * 2 * 32

                    original_bits = total_n * 32
                    if use_quantization:
                        q_bytes = b"".join(q_bytes_chunks)
                        zstd_compressed = compress_zstd(q_bytes, level=22)
                        compressed_bits = len(zstd_compressed) * 8 + metadata_bits
                    else:
                        compressed_bits = original_bits
                    zstd_ratio = compressed_bits / original_bits

                    # Dense entropy diagnostics.
                    dense_entropy_bits = (
                        float(entropy_total)
                        if use_quantization
                        else float(original_bits)
                    )
                    dense_entropy_bits_per_weight = dense_entropy_bits / float(total_n)
                    dense_entropy_ratio = dense_entropy_bits / float(original_bits)
                    metadata_ratio = metadata_bits / float(original_bits)                    

                    # Sparse proxy:
                    # global bitmask + compressed non-zero quantized indices.
                    global_nz_mask = torch.cat([m.reshape(-1) for m in nz_masks])
                    mask_np = global_nz_mask.to(torch.uint8).cpu().numpy()

                    bitmask_bytes = pack_bitmaskGPU(mask_np)
                    compressed_mask = compress_zstd(bitmask_bytes, level=22)

                    mask_compressed_bits = len(compressed_mask) * 8
                    if use_quantization:
                        nz_idx_bytes = b"".join(nz_idx_bytes_chunks)
                        compressed_values = compress_zstd(nz_idx_bytes, level=22)
                        values_compressed_bits = len(compressed_values) * 8
                    else:
                        values_compressed_bits = total_nz * 32

                    compressed_sparse_bits = (
                        mask_compressed_bits +
                        values_compressed_bits +
                        metadata_bits
                    )

                    sparse_ratio = compressed_sparse_bits / original_bits
                    sparsity = 1.0 - float(total_nz) / float(total_n)

                    # Zstd component ratios.
                    mask_zstd_ratio = mask_compressed_bits / float(original_bits)
                    values_zstd_ratio = values_compressed_bits / float(original_bits)

                    # Entropy proxy for the sparse representation:
                    # H(mask) + P(nonzero) * H(bucket | nonzero) + metadata.
                    p_nz = float(total_nz) / float(total_n)
                    p_zero = 1.0 - p_nz

                    mask_entropy_bits_per_weight = 0.0
                    if p_zero > 0.0:
                        mask_entropy_bits_per_weight -= p_zero * math.log2(p_zero)
                    if p_nz > 0.0:
                        mask_entropy_bits_per_weight -= p_nz * math.log2(p_nz)

                    mask_entropy_bits = mask_entropy_bits_per_weight * float(total_n)
                    mask_entropy_ratio = mask_entropy_bits / float(original_bits)

                    if not use_quantization:
                        # Raw FP32 has no codebook-index entropy. Count 32 bits
                        # per surviving value in this conservative proxy.
                        nonzero_index_entropy_bits_per_nonzero = 32.0
                        nonzero_index_entropy_bits = 32.0 * float(total_nz)
                    elif total_nz > 0:
                        nz_probs = nz_index_counts / nz_index_counts.sum().clamp_min(1.0)
                        nz_probs = nz_probs[nz_probs > 0]

                        nonzero_index_entropy_bits_per_nonzero = float(
                            (-(nz_probs * torch.log2(nz_probs)).sum()).item()
                        )
                        nonzero_index_entropy_bits = nonzero_index_entropy_bits_per_nonzero * float(total_nz)
                    else:
                        nonzero_index_entropy_bits_per_nonzero = 0.0
                        nonzero_index_entropy_bits = 0.0

                    nonzero_index_entropy_ratio = nonzero_index_entropy_bits / float(original_bits)

                    sparse_entropy_proxy_bits = (
                        mask_entropy_bits +
                        nonzero_index_entropy_bits +
                        metadata_bits
                    )
                    sparse_entropy_proxy_ratio = sparse_entropy_proxy_bits / float(original_bits)

                    if use_quantization:
                        sparse_symbol_probs = (
                            sparse_symbol_counts
                            / sparse_symbol_counts.sum().clamp_min(1.0)
                        )
                        sparse_symbol_probs = sparse_symbol_probs[sparse_symbol_probs > 0]
                        sparse_symbol_H_bits_per_weight = float(
                            (
                                -(sparse_symbol_probs * torch.log2(sparse_symbol_probs)).sum()
                            ).item()
                        )
                    else:
                        sparse_symbol_H_bits_per_weight = (
                            mask_entropy_bits_per_weight + 32.0 * p_nz
                        )

                    sparse_symbol_H_ratio = sparse_symbol_H_bits_per_weight / 32.0                    

                    zstd_ratios.append(float(zstd_ratio))
            else:
                quantized_entropy = zstd_ratio = sparse_ratio = sparsity = None

            with torch.no_grad():
                _load_flat_params_(flat_q)
                quantized_accuracy = test_accuracyGPU(model, testloader, device)

            if torch.equal(flat_q, flat_s):
                sparse_accuracy = quantized_accuracy
            else:
                with torch.no_grad():
                    _load_flat_params_(flat_s)
                    sparse_accuracy = test_accuracyGPU(model, testloader, device)

            with torch.no_grad():
                _load_flat_params_(w_backup)

            checksum_range = _param_checksum_range() if check_ddp_sync else None
            model_matches_sparse_deploy = torch.equal(w_backup, flat_s)

            if local_rank == 0:
                weighted_l2_norm = (last_loss_grad_norm * perspective_coeff) if last_loss_grad_norm is not None else None
                # last_custom_beta_norm is already the applied (entropy_coeff-scaled) entropy
                # gradient norm.  Report the realized entropy fraction instead of
                # re-multiplying by entropy_coeff (which would be misleading).
                weighted_custom_norm = last_entropy_fraction
                training_time_global = round(time.time() - start_time_global)
                sparse_mask_diag = (
                    _train_sparse_mask_diagnostics()
                    if train_sparse
                    else None
                )
                codebook_tie_error = None
                codebook_min_gap = None
                if train_centroids and codebook_active:
                    with torch.no_grad():
                        tie_errors = []
                        gaps = []
                        for p_idx, param in enumerate(params_for_quant):
                            assignment = fixed_codebook_assignments[p_idx]
                            centroids = v_list[p_idx]
                            tie_errors.append(
                                (
                                    param.detach().reshape(-1)
                                    - centroids[assignment]
                                ).abs().max()
                            )
                            if centroids.numel() > 1:
                                gaps.append((centroids[1:] - centroids[:-1]).min())
                        codebook_tie_error = torch.stack(tie_errors).max().item()
                        codebook_min_gap = torch.stack(gaps).min().item()

                if epoch == 0:
                    log += f"delta = {delta}\n"
                log += (
                    f"Epoch {epoch + 1}: "
                    f"A_NQ = {accuracy}, "
                    f"A_Q = {quantized_accuracy}, H_Q = {quantized_entropy}, "
                    f"zstd_ratio = {zstd_ratio:.2%}, sparse_ratio = {sparse_ratio:.2%}, "
                    f"sparsity = {sparsity:.2%}, sparse_accuracy = {sparse_accuracy}, "
                    f"training_time = {training_time_global}s\n"
                )

                sparse_accuracies.append(sparse_accuracy)
                sparse_ratios.append(float(sparse_ratio) if sparse_ratio is not None else None)

                with torch.no_grad():
                    # Log weight percentiles to understand whether the active
                    # quantization grid covers the useful weight range.
                    w_stats = torch.cat([p.detach().reshape(-1).float() for p in model.parameters()])
                    qvals = _percentiles_large_tensor(w_stats, [0.001, 0.01, 0.5, 0.99, 0.999])

                    p001 = qvals[0].item()
                    p01 = qvals[1].item()
                    p50 = qvals[2].item()
                    p99 = qvals[3].item()
                    p999 = qvals[4].item()         

                    lsq_scale_values = None
                    lsq_total_scale_grad_last = None
                    lsq_clipping_fractions = None
                    if lsq_enabled:
                        # Under per-channel a tensor has one step size per output
                        # channel; the log keeps one number per tensor, so it
                        # reports the mean.
                        lsq_scale_values = [
                            float(s.detach().mean().item()) for s in lsq_scales
                        ]
                        lsq_total_scale_grad_last = [
                            0.0 if s.grad is None else float(s.grad.mean().item())
                            for s in lsq_scales
                        ]
                        lsq_clipping_fractions = []
                        for p_idx, param in enumerate(params_for_quant):
                            scale = lsq_scales[p_idx].detach().clamp_min(1e-12)
                            if lsq_per_channel:
                                scale = expand_scale_flat(scale, param)
                            q_full = lsq_integer_codebooks[p_idx]
                            normalized = param.detach().reshape(-1) / scale
                            clipped = (normalized < q_full[0]) | (
                                normalized > q_full[-1]
                            )
                            lsq_clipping_fractions.append(
                                float(clipped.float().mean().item())
                            )

                delta_debug_log = ""

                if hasattr(FISTA_leonardo, "_delta_debug"):
                    dbg = FISTA_leonardo._delta_debug
                    calls = max(1, dbg["calls"])

                    delta_debug_log = (
                        f"delta_debug: "
                        f"mean_sum_x={dbg['mean_sum_x'] / calls:.6e}, "
                        f"mean_violation={dbg['mean_violation'] / calls:.6e}, "
                        f"frac_null_x={dbg['frac_null_x'] / calls:.6%}, "
                        f"frac_sum_x_lt_0_5={dbg['frac_sum_x_lt_0_5'] / calls:.6%}, "
                        f"frac_two_bucket={dbg['frac_two_bucket'] / calls:.6%}, "
                        f"xi_zero={dbg.get('xi_zero', 0.0) / calls:.6e}, "
                        f"xi_bucket_mean={dbg.get('xi_bucket_mean', 0.0) / calls:.6e}, "
                        f"xi_bucket_min={dbg.get('xi_bucket_min', 0.0) / calls:.6e}, "
                        f"xi_bucket_max={dbg.get('xi_bucket_max', 0.0) / calls:.6e}, "
                        f"effective_xi_mean={dbg.get('effective_xi_mean', 0.0) / calls:.6e}, "
                        f"effective_xi_min={dbg.get('effective_xi_min', 0.0) / calls:.6e}, "
                        f"effective_xi_max={dbg.get('effective_xi_max', 0.0) / calls:.6e}, "
                        f"objective_constant={dbg.get('objective_constant', 0.0) / calls:.6e}, "
                        f"sum_z_per_weight={dbg.get('sum_z_per_weight', 0.0) / calls:.6e}, "
                        f"c_zero_per_weight={dbg.get('c_zero_per_weight', 0.0) / calls:.6e}, "
                        f"g_zero_per_weight={dbg.get('g_zero_per_weight', 0.0) / calls:.6e}"
                    )

                    FISTA_leonardo._delta_debug = {
                        "calls": 0,
                        "mean_sum_x": 0.0,
                        "mean_violation": 0.0,
                        "frac_null_x": 0.0,
                        "frac_sum_x_lt_0_5": 0.0,
                        "frac_two_bucket": 0.0,

                        "xi_zero": 0.0,
                        "xi_bucket_mean": 0.0,
                        "xi_bucket_min": 0.0,
                        "xi_bucket_max": 0.0,
                        "effective_xi_mean": 0.0,
                        "effective_xi_min": 0.0,
                        "effective_xi_max": 0.0,
                        "objective_constant": 0.0,
                        "sum_z_per_weight": 0.0,
                        "c_zero_per_weight": 0.0,
                        "g_zero_per_weight": 0.0,
                    }                        

                print(f"============== Epoch {epoch + 1}/{n_epochs} ==============", flush=True)
                print(f"train_batches = {train_batches}", flush=True)
                print(f"training_time_without_metrics = {training_time_without_metrics}s", flush=True)
                print(f"perspective_coeff = {perspective_coeff}", flush=True)
                print(f"entropy_coeff_current = {entropy_coeff_current}", flush=True)
                print(f"flat_schedule = {flat_schedule}", flush=True)
                print(f"exposure_epoch = {exposure_epoch:.6f}, exposure_cum = {exposure_cum:.6f}", flush=True)
                if prox_diag_count > 0:
                    print(
                        f"prox_diag: gamma={prox_gamma}, applications={prox_diag_count}, "
                        f"displacement_mean={prox_disp_sum / prox_diag_count:.6e}, "
                        f"zero_frac_mean={prox_zero_sum / prox_diag_count:.6f}",
                        flush=True
                    )
                print(f"entropy_every = {entropy_every}", flush=True)
                print(f"entropy_steps = {entropy_steps}", flush=True)
                if lsq_enabled:
                    print(
                        "lsq_diag: "
                        f"joint_metaq={joint_lsq_metaq}, "
                        f"scale_lr={(scale_optimizer.param_groups[0]['lr'] if scale_optimizer is not None else lsq_scale_lr):.6e}, "
                        f"scales={lsq_scale_values}, "
                        f"task_scale_grad_last={lsq_task_scale_grad_last}, "
                        f"metaq_scale_grad_last={lsq_metaq_scale_grad_last}, "
                        f"total_scale_grad_last={lsq_total_scale_grad_last}, "
                        f"clipping_fractions={lsq_clipping_fractions}",
                        flush=True,
                    )
                if persp_diag_count > 0:
                    n_d = persp_diag_count
                    persp_entropy_fraction = (
                        persp_entropy_norm_sum / max(persp_grad_norm_sum, 1e-12)
                    )
                    print(
                        f"perspective_entropy_diag: "
                        f"applied_entropy_norm_mean={persp_entropy_norm_sum / n_d:.6e}, "
                        f"grad_norm_mean={persp_grad_norm_sum / n_d:.6e}, "
                        f"entropy_fraction={persp_entropy_fraction:.6f}, "
                        f"beta_commonmode_mean={persp_commonmode_sum / n_d:.6f}, "
                        f"xi_min={persp_xi_min:.6e}, xi_max={persp_xi_max:.6e}, "
                        f"xi_mean={persp_xi_mean_sum / n_d:.6e}, "
                        f"xi_pinned_frac={persp_xi_pinned_sum / n_d:.6f}, "
                        f"dual_step={dual_step}",
                        flush=True
                    )
                if last_loss_grad_norm is not None:
                    print(f"loss_grad_norm_last_batch = {last_loss_grad_norm:.6e}", flush=True)
                    print(f"weighted_l2_norm_last_batch = {weighted_l2_norm:.6e}", flush=True)
                if last_custom_beta_norm is not None:
                    print(f"custom_beta_norm_last_applied = {last_custom_beta_norm:.6e}", flush=True)
                    print(f"weighted_custom_norm_last_applied = {weighted_custom_norm:.6e}", flush=True)
                if checksum_range is not None:
                    print(f"ddp_param_checksum_range = {checksum_range:.6e}", flush=True)
                if sparse_mask_diag is not None:
                    masked_count, violation_count, violation_max_abs = sparse_mask_diag
                    print(
                        "train_sparse_mask_diag: "
                        f"masked_count={masked_count}, "
                        f"nonzero_violation_count={violation_count}, "
                        f"max_abs={violation_max_abs:.6e}, "
                        f"model_matches_sparse_deploy={model_matches_sparse_deploy}",
                        flush=True,
                    )
                if codebook_tie_error is not None:
                    centroid_grad_diag = ""
                    if last_centroid_grad_norm is not None:
                        centroid_grad_diag = (
                            f", centroid_grad_norm_last_batch="
                            f"{last_centroid_grad_norm:.6e}, "
                            f"centroid_grad_abs_max_last_batch="
                            f"{last_centroid_grad_abs_max:.6e}"
                        )
                    displacement_mean = (
                        centroid_displacement_mean_sum
                        / max(1, centroid_displacement_count)
                    )
                    print(
                        f"trainable_centroids_diag: "
                        f"lr_scale={centroid_lr_scale:.6e}, "
                        f"max_tie_error={codebook_tie_error:.6e}, "
                        f"min_sorted_gap={codebook_min_gap:.6e}, "
                        f"displacement_max_epoch={centroid_displacement_max:.6e}, "
                        f"displacement_mean_per_layer_step={displacement_mean:.6e}"
                        f"{centroid_grad_diag}",
                        flush=True,
                    )
                print(f"A_NQ = {accuracy}", flush=True)
                print(f"A_Q = {quantized_accuracy}", flush=True)
                print(
                    "representation_mode = "
                    + ("quantized_indices" if use_quantization else "fp32_pruning_only"),
                    flush=True,
                )
                print(f"H_Q = {quantized_entropy}", flush=True)
                print(f"zstd_ratio = {zstd_ratio:.2%}", flush=True)
                print(f"sparse_ratio = {sparse_ratio:.2%}", flush=True)
                print(f"sparsity = {sparsity:.2%}", flush=True)
                if z_mask_total > 0:
                    print(
                        f"z_mask_fraction = {z_mask_pruned / z_mask_total:.2%}, "
                        f"encoded_zero_fraction = {sparsity:.2%}",
                        flush=True,
                    )
                print(f"sparse_accuracy = {sparse_accuracy}", flush=True)
                print(f"dual_zero_rounding = {dual_zero_rounding}", flush=True)
                print(f"dual_zero_score_eps = {dual_zero_score_eps}", flush=True)
                print(f"dual_zero_abs_power = {dual_zero_abs_power}", flush=True)
                print(f"dual_zero_candidate_multiplier = {dual_zero_candidate_multiplier}", flush=True)
                print(
                    f"dense_entropy_debug: "
                    f"H_Q_bits_per_weight={dense_entropy_bits_per_weight:.6f}, "
                    f"H_Q_ratio={dense_entropy_ratio:.4%}",
                    flush=True
                )
                print(
                    f"sparse_zstd_components: "
                    f"mask_zstd_ratio={mask_zstd_ratio:.4%}, "
                    f"values_zstd_ratio={values_zstd_ratio:.4%}, "
                    f"metadata_ratio={metadata_ratio:.6%}",
                    flush=True
                )
                print(
                    f"sparse_entropy_proxy: "
                    f"mask_H_bits_per_weight={mask_entropy_bits_per_weight:.6f}, "
                    f"nonzero_index_H_bits_per_nonzero={nonzero_index_entropy_bits_per_nonzero:.6f}, "
                    f"mask_H_ratio={mask_entropy_ratio:.4%}, "
                    f"nonzero_index_H_ratio={nonzero_index_entropy_ratio:.4%}, "
                    f"sparse_proxy_ratio={sparse_entropy_proxy_ratio:.4%}, "
                    f"sparse_symbol_H_bits_per_weight={sparse_symbol_H_bits_per_weight:.6f}, "
                    f"sparse_symbol_H_ratio={sparse_symbol_H_ratio:.4%}",
                    flush=True
                )
                print(
                    f"weight_percentiles: "
                    f"p0.1={p001:.6e}, p1={p01:.6e}, "
                    f"p50={p50:.6e}, p99={p99:.6e}, p99.9={p999:.6e}",
                    flush=True
                )                
                print(f"current lr = {optimizer.param_groups[0]['lr']}", flush=True)
                if distill_loss_last is not None:
                    print(
                        f"distill_diag: alpha={distill_alpha}, tau={distill_tau}, "
                        f"task_loss_last={task_loss_last:.6f}, "
                        f"distill_loss_last={distill_loss_last:.6f}",
                        flush=True,
                    )
                print(f"training_time = {training_time_global}s", flush=True)
                # The running histories printed each epoch are the DELIVERED
                # metrics: the accuracy of the sparse quantized model actually
                # shipped, and the size of its serialized package. The dense
                # zstd ratio and the unquantized accuracy remain available above
                # in this same block, per epoch.
                print(f"sparse_accuracies = {sparse_accuracies}", flush=True)
                print(f"sparse_ratios = {sparse_ratios}", flush=True)
                if z_recompute_count > 0:
                    recompute_frac = (z_recompute_sum / z_recompute_count).item() * 100.0
                    print(
                        f"z_recompute_frac (z>0.5 via _z_prune_mask right after dual) = {recompute_frac:.2f}%",
                        flush=True
                    )
                if delta_debug_log:
                    print(delta_debug_log, flush=True)
                if use_perspective:
                    with torch.no_grad():
                        y_means, z_means, z_fracs, z_p10s, z_p50s, z_p90s = [], [], [], [], [], []
                        prune_fracs, thr_list = [], []
                        n_tot, n_pruned = 0, 0
                        for p_idx, param in enumerate(params_for_quant):
                            wl = param.detach().reshape(-1)
                            vl = v_list[p_idx]
                            thr = target_prune_thresholds[p_idx]
                            y_star_diag = _perspective_y_star(wl, vl)
                            z_diag = (1.0 - y_star_diag).clamp(0.0, 1.0)
                            y_means.append(y_star_diag.mean().item())
                            z_means.append(z_diag.mean().item())
                            z_fracs.append((z_diag > z_prune_threshold).float().mean().item())
                            # torch.quantile does not support tensors of AlexNet
                            # scale on Leonardo; use the existing bounded-memory
                            # percentile helper instead.
                            z_q = _percentiles_large_tensor(z_diag, [0.10, 0.50, 0.90])
                            z_p10s.append(z_q[0].item())
                            z_p50s.append(z_q[1].item())
                            z_p90s.append(z_q[2].item())
                            pm = _mask_to_apply(p_idx, wl, vl)   # test_144: frozen if on
                            prune_fracs.append(pm.float().mean().item())
                            eff_thr = thr if thr is not None else (mag_prune_ratio * vl.abs().min())
                            thr_list.append(float(eff_thr))
                            n_tot += wl.numel()
                            n_pruned += int(pm.sum().item())
                        mean_y = sum(y_means) / max(1, len(y_means))
                        mean_z = sum(z_means) / max(1, len(z_means))
                        mean_z_frac = sum(z_fracs) / max(1, len(z_fracs))
                        overall_prune = n_pruned / max(1, n_tot)
                        l1_push = 2.0 * math.sqrt(perspective_coeff * sparsity_coeff) if sparsity_coeff > 0 else 0.0
                        print(
                            f"perspective_debug: perspective_coeff={perspective_coeff:.3e}, sparsity_coeff={sparsity_coeff:.3e}, "
                            f"target_sparsity={target_sparsity}, mag_prune_ratio={mag_prune_ratio:.3f}, "
                            f"l1_push_near_zero={l1_push:.3e}, "
                            f"mean_y_star={mean_y:.4f}, mean_z_star={mean_z:.4f}, "
                            f"z_gt_0.5_frac_mean_layers={mean_z_frac:.4%}, "
                            f"z_quantiles_mean_layers=[{sum(z_p10s)/max(1,len(z_p10s)):.4f},"
                            f"{sum(z_p50s)/max(1,len(z_p50s)):.4f},"
                            f"{sum(z_p90s)/max(1,len(z_p90s)):.4f}], "
                            f"overall_prune_frac={overall_prune:.4%}, "
                            f"mag_thr_first_layers={[round(t, 5) for t in thr_list[:4]]}, "
                            f"prune_frac_first_layers={[round(f, 4) for f in prune_fracs[:4]]}",
                            flush=True
                        )
                print("====================================\n", flush=True)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            _dist_barrier()

            if adiabatic_enabled:
                previous_progress = adiabatic_progress
                if local_rank == 0:
                    adiabatic_point = {
                        "epoch": epoch + 1,
                        "progress": previous_progress,
                        "sparse_accuracy": float(sparse_accuracy),
                        "sparsity": float(sparsity),
                        "sparse_ratio": float(sparse_ratio),
                    }
                    if (
                        float(sparse_accuracy) >= adiabatic_accuracy_target
                        and (
                            adiabatic_best_target is None
                            or previous_progress > adiabatic_best_target["progress"]
                        )
                    ):
                        adiabatic_best_target = adiabatic_point
                    if (
                        float(sparse_accuracy)
                        >= adiabatic_accuracy_target - adiabatic_accuracy_tolerance
                        and (
                            adiabatic_best_floor is None
                            or previous_progress > adiabatic_best_floor["progress"]
                        )
                    ):
                        adiabatic_best_floor = adiabatic_point
                (
                    adiabatic_progress,
                    adiabatic_good_epochs,
                    adiabatic_action,
                ) = _update_adiabatic_progress(
                    progress=adiabatic_progress,
                    good_epochs=adiabatic_good_epochs,
                    sparse_accuracy=float(sparse_accuracy),
                    target_accuracy=float(adiabatic_accuracy_target),
                    tolerance=float(adiabatic_accuracy_tolerance),
                    step=float(adiabatic_step),
                    backoff=float(adiabatic_backoff),
                    patience=int(adiabatic_patience),
                )
                if local_rank == 0:
                    print(
                        f"[ADIABATIC CONTROL] epoch={epoch + 1}, "
                        f"sparse_accuracy={float(sparse_accuracy):.4f}, "
                        f"target={adiabatic_accuracy_target:.4f}, "
                        f"floor={adiabatic_accuracy_target - adiabatic_accuracy_tolerance:.4f}, "
                        f"action={adiabatic_action}, progress={previous_progress:.4f}"
                        f"->{adiabatic_progress:.4f}, good_epochs={adiabatic_good_epochs}",
                        flush=True,
                    )

            if model_name in (
                "AlexNet",
                "VGG16",
                "ResNet-18",
                "ResNet-50",
                "DeiT-Small",
            ) and accuracy < 0.12 and epoch >= 3:
                if local_rank == 0:
                    log += f"Accuracy is too low! (A1.0), delta: {delta}\n"
                    log += "-" * 60
                    print(log, flush=True)
                return

            if (model_name == "LeNet5" or model_name == "LeNet300_100") and accuracy < 12 and epoch >= 3:
                if local_rank == 0:
                    log += f"Accuracy is too low! (A1.0), delta: {delta}\n"
                    log += "-" * 60
                    print(log, flush=True)
                return

        gc.collect()

    log += "-" * 60
    if local_rank == 0:
        print(log, flush=True)
        print(f"accuracies = {accuracies}", flush=True)
        print(f"zstd_ratios = {zstd_ratios}", flush=True)
        print(f"sparse_accuracies = {sparse_accuracies}", flush=True)
        print(f"sparse_ratios = {sparse_ratios}", flush=True)
        if adiabatic_enabled:
            print(f"adiabatic_best_target = {adiabatic_best_target}", flush=True)
            print(f"adiabatic_best_floor = {adiabatic_best_floor}", flush=True)
        checkpoint_path = os.environ.get("METAQ_CHECKPOINT_PATH")
        if checkpoint_path:
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            state_model = model.module if hasattr(model, "module") else model
            checkpoint = {
                "format": "metaq_training_v2",
                "epoch": int(n_epochs),
                "model_fp32": state_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scale_optimizer": scale_optimizer.state_dict() if scale_optimizer is not None else None,
                "scale_scheduler": scale_scheduler.state_dict() if scale_scheduler is not None else None,
                "config": {
                    "model_name": model_name,
                    "C": C,
                    "quantizer": quantizer,
                    "lsq_per_channel": lsq_per_channel,
                    "entropy_coeff": entropy_coeff,
                    "sparsity_coeff": sparsity_coeff,
                },
            }
            tmp_path = checkpoint_path + ".tmp"
            torch.save(checkpoint, tmp_path)
            os.replace(tmp_path, checkpoint_path)
            print(f"[CHECKPOINT SAVED] {checkpoint_path}", flush=True)
    return
