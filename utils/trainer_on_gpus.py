import torch
import os
import time 
import numpy as np
import copy
import struct
import sys
import torch.optim as optim
import torch.distributed as dist
import gc
from utils.quantize_and_compress import compute_entropy, quantize_weights_center, compute_entropyGPU, quantize_weights_centerGPU, compute_entropy_hist
from utils.optimization import FISTA, FISTA_leonardo, ProximalBM, test_accuracy, test_accuracyGPU
from utils.weight_utils import initialize_weights
from utils.quantize_and_compress import compress_zstd, BestQuantization, pack_bitmask, pack_bitmaskGPU
from datetime import datetime, timedelta

def train_and_evaluate(model, model_name, criterion, C, lr, lambda_reg, alpha, T1_explicit, T2_explicit, subgradient_step, w0, r, 
                       first_best_indices, BestQuantization_target_acc, final_target_acc, target_zstd_ratio, min_xi, max_xi, upper_c, 
                       lower_c, c1, c2, zeta, l, n_epochs, max_iterations, device, train_optimizer, entropy_optimizer, trainloader,
                       testloader, train_sampler, steps_per_epoch, delta, pruning, QuantizationType, sparsity_threshold, accuracy_tollerance,
                       metrics_interval=1, entropy_warmup_epochs=0, entropy_every=1, check_ddp_sync=False):
    """Train and evaluate a model with optional entropy regularization.

    This function is intentionally self-contained because the compression
    experiments require tight control over the training loop.  The regular
    cross-entropy gradient is produced by `loss.backward()` and synchronized by
    DDP.  The entropy gradient is then added manually; therefore, when DDP is
    active, we explicitly all-reduce the final gradients before `optimizer.step()`.

    Args related to entropy:
        T2_explicit: final weight of the entropy regularizer.  If zero, FISTA is
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
    """

    torch.set_num_threads(1)

    local_rank = dist.get_rank() if dist.is_initialized() else 0

    if device.type == "cuda":
        torch.cuda.set_device(device.index if device.index is not None else local_rank)

    if train_optimizer == 'ADAM':
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=T1_explicit)
    elif train_optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=T1_explicit)
    else:
        raise ValueError(f"Unsupported optimizer: {train_optimizer}")

    # Initial quantization grid.  This may later be replaced by an adaptive
    # percentile-based grid at the first entropy step.
    min_w, max_w = w0 - r, w0 + r
    v = torch.linspace(min_w, max_w - (max_w - min_w) / C, steps=C, device=device)

    # IMPORTANT: every rank must use the same xi. Otherwise the custom gradient differs
    # after DDP has already synchronized the loss gradient.
    xi = min_xi + (max_xi - min_xi) * torch.rand(C, device=device)
    xi = torch.sort(xi)[0]
    if dist.is_initialized():
        dist.broadcast(xi, src=0)

    log = ""
    accuracy = None
    accuracies, entropies = [], []
    global_step = 0

    # The adaptive grid must be reset only once for the whole run, not once per
    # epoch.  It is triggered at the first real entropy step, so it remains
    # correct even when `global_step % entropy_every != 0` at batch 0.
    grid_reset_done = False

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

    for epoch in range(n_epochs):
        should_eval_epoch = ((epoch + 1) % metrics_interval == 0) or (epoch == n_epochs - 1)

        # T2 schedule: no entropy during warmup, then a gentle exponential ramp
        # from T2/8 to T2.  This avoids abruptly injecting a large custom
        # gradient after many epochs of standard training.
        if T2_explicit > 0 and epoch >= entropy_warmup_epochs:
            t = epoch - entropy_warmup_epochs
            T2_current = T2_explicit * (1.0 - np.exp(-t)) + (T2_explicit / 8.0) * np.exp(-t)
        else:
            T2_current = 0.0

        for param_group in optimizer.param_groups:
            param_group['weight_decay'] = T1_explicit

        if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        start_time_global = time.time()
        train_batches = 0
        entropy_steps = 0
        last_loss_grad_norm = None
        last_custom_beta_norm = None

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

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=(device.type == "cuda")
            ):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            loss.backward()

            is_last_configured_step = steps_per_epoch is not None and (i + 1) >= steps_per_epoch
            capture_last_norms = local_rank == 0 and should_eval_epoch and is_last_configured_step
            if capture_last_norms:
                last_loss_grad_norm = _grad_norm_from_current_grads()

            beta_tensor = None
            # Entropy/FISTA is the expensive part of the method.  Applying it
            # every N global steps keeps the computational cost controlled.
            apply_entropy = T2_current > 0 and (global_step % entropy_every == 0)
            if apply_entropy:
                if not grid_reset_done:
                    # Build a tighter quantization grid from the current weight
                    # distribution.  The fixed default grid can be too wide for
                    # trained AlexNet weights, making C=16 quantization collapse
                    # accuracy.  Percentiles avoid extreme outliers.
                    with torch.no_grad():
                        w_for_grid = torch.cat([p.detach().reshape(-1).float() for p in model.parameters()])
                        # p0.1 and p99.9 define the adaptive grid range.
                        qs = torch.tensor([0.001, 0.999], device=device)
                        lo, hi = torch.quantile(w_for_grid, qs)

                        w0 = ((lo + hi) / 2).item()
                        r = ((hi - lo) / 2).item()

                        v = torch.linspace(
                            w0 - r,
                            w0 + r - (2 * r) / C,
                            steps=C,
                            device=device
                        )

                        if dist.is_initialized():
                            dist.broadcast(v, src=0)

                        if local_rank == 0:
                            print(
                                f"[GRID RESET] epoch={epoch + 1}, "
                                f"w0={w0:.6e}, r={r:.6e}, "
                                f"lo={lo.item():.6e}, hi={hi.item():.6e}",
                                flush=True
                            )  
                    grid_reset_done = True
                                  
                with torch.no_grad():
                    # FISTA works on the current flat parameter vector and
                    # returns beta multipliers with one entry per parameter.
                    w = torch.cat([param.detach().reshape(-1) for param in model.parameters()]).to(device)
                    zeta *= 1 + l
                    l = l / 1.5

                    if entropy_optimizer == 'FISTA':
                        xi, beta_tensor = FISTA_leonardo(
                            xi, v, w, C, upper_c, lower_c, delta,
                            subgradient_step, device, max_iterations, pruning
                        )
                    elif entropy_optimizer == 'PROXIMAL BM':
                        xi, beta_tensor = ProximalBM(
                            xi, v, w, C, upper_c, lower_c, delta,
                            zeta, subgradient_step, device, max_iterations, pruning
                        )
                    else:
                        raise ValueError(f"Unsupported entropy optimizer: {entropy_optimizer}")

                    # Add the entropy gradient to the already-computed loss
                    # gradient.  The sign convention follows the existing METaQ
                    # formulation: the beta multipliers enter with a minus sign.
                    idx = 0
                    for param in model.parameters():
                        numel = param.numel()
                        if param.grad is not None:
                            update = (T2_current * (-beta_tensor[idx:idx + numel])).view(param.size())
                            param.grad.add_(update)
                        idx += numel

                    if dist.is_initialized():
                        # DDP synchronized only the loss gradient inside
                        # loss.backward().  Since the entropy term is added
                        # afterwards, we must synchronize the final gradients
                        # manually before optimizer.step().
                        for param in model.parameters():
                            if param.grad is not None:
                                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                                param.grad.div_(dist.get_world_size())

                        dist.broadcast(xi, src=0)                        

                    entropy_steps += 1
                    if local_rank == 0:
                        last_custom_beta_norm = beta_tensor.float().norm().item()

            optimizer.step()

        training_time_without_metrics = round(time.time() - start_time_global)

        if should_eval_epoch:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            _dist_barrier()

            with torch.no_grad():
                accuracy = test_accuracyGPU(model, testloader, device)
            if local_rank == 0:
                accuracies.append(accuracy)

            _dist_barrier()

            with torch.no_grad():
                # Backup float weights, then evaluate quantized/sparse variants
                # in-place.  The original weights are restored after metrics.
                w_backup = torch.cat([p_.detach().reshape(-1) for p_ in model.parameters()]).clone()

                if QuantizationType == "center":
                    v_centers = (v[:-1] + v[1:]) / 2
                    v_centers = torch.cat([v_centers, v[-1:]])
                    flat_q = quantize_weights_centerGPU(w_backup, v, v_centers, device=device)
                else:
                    flat_q = w_backup.clone()

                flat_s = flat_q.clone()
                flat_s[flat_s.abs() <= sparsity_threshold] = 0.0

            if local_rank == 0:
                with torch.no_grad():
                    # Quantized index stream used for entropy and zstd metrics.
                    # This is an index-only compression proxy, not by itself a
                    # deployable Deep-Compression-style model format.
                    q_idx = (torch.bucketize(w_backup, v, right=False) - 1).clamp_(0, C - 1)

                    counts = torch.bincount(q_idx, minlength=C).to(torch.float32)
                    p = counts / counts.sum()
                    p = p[p > 0]
                    quantized_entropy = float((-(p * torch.log2(p)).sum() * counts.sum()).item())
                    quantized_entropy = round(quantized_entropy) + 1
                    entropies.append(quantized_entropy)

                    if C <= 256:
                        dtype = torch.uint8
                        index_bits = 8
                    elif C <= 65536:
                        dtype = torch.uint16
                        index_bits = 16
                    else:
                        dtype = torch.int32
                        index_bits = 32

                    # Compress the quantized indices with zstd.  The ratio is
                    # measured against the FP32 model size.
                    q_bytes = q_idx.to(dtype).cpu().numpy().tobytes()
                    zstd_compressed = compress_zstd(q_bytes, level=22)
                    N = q_idx.numel()
                    original_bits = N * 32
                    metadata_bits = index_bits + 32 + 32
                    compressed_bits = (len(zstd_compressed) * 8) + metadata_bits
                    zstd_ratio = compressed_bits / original_bits

                    v_centers = (v[:-1] + v[1:]) / 2
                    v_centers = torch.cat([v_centers, v[-1:]])
                    q_vals = v_centers[q_idx]
                    # Sparse proxy: keep a bitmask plus compressed non-zero
                    # quantized indices.
                    nz_mask = (q_vals.abs() > sparsity_threshold)
                    mask_np = nz_mask.to(torch.uint8).cpu().numpy()
                    bitmask_bytes = pack_bitmaskGPU(mask_np)
                    nz_idx_bytes = q_idx[nz_mask].to(dtype).cpu().numpy().tobytes()
                    compressed_mask = compress_zstd(bitmask_bytes, level=22)
                    compressed_values = compress_zstd(nz_idx_bytes, level=22)
                    compressed_sparse_bits = (
                        len(compressed_mask) * 8 +
                        len(compressed_values) * 8 +
                        metadata_bits
                    )
                    sparse_ratio = compressed_sparse_bits / original_bits
                    sparsity = 1.0 - float(mask_np.sum()) / float(mask_np.size)
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

            if local_rank == 0:
                weighted_l2_norm = (last_loss_grad_norm * T1_explicit) if last_loss_grad_norm is not None else None
                weighted_custom_norm = (last_custom_beta_norm * T2_current) if last_custom_beta_norm is not None else None
                training_time_global = round(time.time() - start_time_global)

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

                with torch.no_grad():
                    # Log weight percentiles to understand whether the active
                    # quantization grid covers the useful weight range.
                    w_stats = torch.cat([p.detach().reshape(-1).float() for p in model.parameters()])
                    qs = torch.tensor([0.001, 0.01, 0.5, 0.99, 0.999], device=w_stats.device)
                    qvals = torch.quantile(w_stats, qs)

                    p001 = qvals[0].item()
                    p01 = qvals[1].item()
                    p50 = qvals[2].item()
                    p99 = qvals[3].item()
                    p999 = qvals[4].item()                

                print(f"============== Epoch {epoch + 1}/{n_epochs} ==============", flush=True)
                print(f"train_batches = {train_batches}", flush=True)
                print(f"training_time_without_metrics = {training_time_without_metrics}s", flush=True)
                print(f"T1 = {T1_explicit}", flush=True)
                print(f"T2_current = {T2_current}", flush=True)
                print(f"entropy_every = {entropy_every}", flush=True)
                print(f"entropy_steps = {entropy_steps}", flush=True)
                if last_loss_grad_norm is not None:
                    print(f"loss_grad_norm_last_batch = {last_loss_grad_norm:.6e}", flush=True)
                    print(f"weighted_l2_norm_last_batch = {weighted_l2_norm:.6e}", flush=True)
                if last_custom_beta_norm is not None:
                    print(f"custom_beta_norm_last_applied = {last_custom_beta_norm:.6e}", flush=True)
                    print(f"weighted_custom_norm_last_applied = {weighted_custom_norm:.6e}", flush=True)
                if checksum_range is not None:
                    print(f"ddp_param_checksum_range = {checksum_range:.6e}", flush=True)
                print(f"A_NQ = {accuracy}", flush=True)
                print(f"A_Q = {quantized_accuracy}", flush=True)
                print(f"H_Q = {quantized_entropy}", flush=True)
                print(f"zstd_ratio = {zstd_ratio:.2%}", flush=True)
                print(f"sparse_ratio = {sparse_ratio:.2%}", flush=True)
                print(f"sparsity = {sparsity:.2%}", flush=True)
                print(f"sparse_accuracy = {sparse_accuracy}", flush=True)
                print(
                    f"weight_percentiles: "
                    f"p0.1={p001:.6e}, p1={p01:.6e}, "
                    f"p50={p50:.6e}, p99={p99:.6e}, p99.9={p999:.6e}",
                    flush=True
                )                
                print(f"training_time = {training_time_global}s", flush=True)
                print("====================================\n", flush=True)                

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            _dist_barrier()

            if (model_name == "AlexNet" or model_name == "VGG16") and accuracy < 0.12 and epoch >= 3:
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
    return
