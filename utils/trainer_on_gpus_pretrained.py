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

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=10,
        gamma=0.1
    )

    # Per-parameter-tensor quantization state.
    # Each layer/tensor gets its own quantization grid v and its own xi.
    # C is still shared, so C=16 still means 16 levels per tensor, i.e. INT4.
    # All parameters are still needed to rebuild the full flat model.
    all_params = list(model.parameters())

    # Only tensors with ndim > 1 are quantized/regularized.
    # For AlexNet this means convolutional and linear weights.
    # Bias tensors are kept in floating point.
    quant_param_indices = [idx for idx, p in enumerate(all_params) if p.ndim > 1]
    params_for_quant = [all_params[idx] for idx in quant_param_indices]
    num_param_tensors = len(params_for_quant)

    min_w, max_w = w0 - r, w0 + r
    v_init = torch.linspace(min_w, max_w - (max_w - min_w) / C, steps=C, device=device)

    v_list = [v_init.clone() for _ in params_for_quant]

    xi_list = []
    for _ in params_for_quant:
        xi_layer = min_xi + (max_xi - min_xi) * torch.rand(C, device=device)
        xi_layer = torch.sort(xi_layer)[0]

        if dist.is_initialized():
            dist.broadcast(xi_layer, src=0)

        xi_list.append(xi_layer)

    log = ""
    accuracy = None
    accuracies, entropies, zstd_ratios = [], [], []
    global_step = 0

    # The adaptive grid must be reset only once for the whole run, not once per
    # epoch.  It is triggered at the first real entropy step, so it remains
    # correct even when `global_step % entropy_every != 0` at batch 0.
    grid_reset_done = False

    # Fraction of each layer range used as explicit zero dead-zone.
    # If |w| <= deadzone_ratio * r_layer, the weight is quantized to exactly zero.
    deadzone_ratio = 0.05

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

                v_layer = v_list[q_state]
                w_flat = param.data.reshape(-1)

                _, q_flat = _quantize_with_deadzone(w_flat, v_layer)

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

    def _quant_centers_from_edges(v_layer: torch.Tensor) -> torch.Tensor:
        """
        Builds quantization centers from bin edges and forces one center to be
        exactly zero. This allows INT4 quantization to create real zero weights.
        """
        centers = (v_layer[:-1] + v_layer[1:]) / 2
        centers = torch.cat([centers, v_layer[-1:]])

        zero_idx = torch.argmin(centers.abs())
        centers[zero_idx] = 0.0

        return centers

    def _quantize_with_deadzone(w_flat: torch.Tensor, v_layer: torch.Tensor):
        """
        Quantizes a flat tensor with per-layer INT4 centers and an explicit
        zero dead-zone around zero.
        """
        centers = _quant_centers_from_edges(v_layer)
        zero_idx = torch.argmin(centers.abs())

        if v_layer.numel() > 1:
            step = v_layer[1] - v_layer[0]
            lo = v_layer[0]
            hi = v_layer[-1] + step
            r_layer = 0.5 * (hi - lo).abs()
        else:
            r_layer = v_layer[0].abs()

        zero_threshold = deadzone_ratio * r_layer

        q_idx = (torch.bucketize(w_flat, v_layer, right=False) - 1).clamp_(0, C - 1)
        q_idx = torch.where(w_flat.abs() <= zero_threshold, zero_idx, q_idx)

        q_flat = centers[q_idx]
        return q_idx, q_flat

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

            loss.backward()

            _restore_fake_quantized_weights_(fake_quant_backups)

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
                    # Build one adaptive quantization grid per parameter tensor.
                    # This keeps C fixed but lets each layer/tensor have its own scale.
                    with torch.no_grad():
                        new_v_list = []
                        grid_infos = []

                        for p_idx, param in enumerate(params_for_quant):
                            w_layer = param.detach().reshape(-1).float()

                            lo, hi = _percentiles_large_tensor(w_layer, [0.001, 0.999])

                            # Avoid degenerate grids for nearly constant tensors.
                            if (hi - lo).abs() < 1e-12:
                                center = 0.5 * (lo + hi)
                                lo = center - 1e-6
                                hi = center + 1e-6

                            w0_layer = ((lo + hi) / 2).item()
                            r_layer = ((hi - lo) / 2).item()

                            v_layer = torch.linspace(
                                w0_layer - r_layer,
                                w0_layer + r_layer - (2 * r_layer) / C,
                                steps=C,
                                device=device
                            )

                            if dist.is_initialized():
                                dist.broadcast(v_layer, src=0)

                            new_v_list.append(v_layer)
                            grid_infos.append((p_idx, w_layer.numel(), lo.item(), hi.item()))

                        v_list = new_v_list

                        if local_rank == 0:
                            first = grid_infos[:4]
                            print(
                                f"[GRID RESET PER-LAYER] epoch={epoch + 1}, "
                                f"num_tensors={len(v_list)}, first_tensors={first}",
                                flush=True
                            )

                    grid_reset_done = True

                with torch.no_grad():
                    # FISTA is applied independently to each parameter tensor.
                    # Each tensor has its own xi and v, but the same C.
                    zeta *= 1 + l
                    l = l / 1.5

                    beta_norm_sq = torch.zeros((), device=device)

                    for p_idx, param in enumerate(params_for_quant):
                        if param.grad is None:
                            continue

                        w_layer = param.detach().reshape(-1).to(device)

                        # upper_c should be local because c_star represents
                        # bucket occupancies for this tensor, not for the whole network.
                        upper_c_layer = float(w_layer.numel())

                        if entropy_optimizer == 'FISTA':
                            xi_list[p_idx], beta_layer = FISTA_leonardo(
                                xi_list[p_idx],
                                v_list[p_idx],
                                w_layer,
                                C,
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
                                C,
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

                        update = (T2_current * (-beta_layer)).view_as(param)
                        param.grad.add_(update)

                        beta_norm_sq += beta_layer.float().pow(2).sum()

                    if dist.is_initialized():
                        # DDP synchronized only the loss gradient inside
                        # loss.backward().  Since the entropy term is added
                        # afterwards, we must synchronize the final gradients
                        # manually before optimizer.step().
                        for param in model.parameters():
                            if param.grad is not None:
                                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                                param.grad.div_(dist.get_world_size())

                        for xi_layer in xi_list:
                            dist.broadcast(xi_layer, src=0)

                    entropy_steps += 1

                    if local_rank == 0:
                        last_custom_beta_norm = torch.sqrt(beta_norm_sq).item()

            optimizer.step()
        
        scheduler.step()

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
                # Backup the full model.
                # Quantize only weight tensors; keep bias tensors in FP32.
                w_backup_parts = []
                flat_q_parts = []
                flat_s_parts = []

                q_idx_layers = []
                q_vals_layers = []

                quant_state = 0

                for full_idx, p_ in enumerate(all_params):
                    w_layer = p_.detach().reshape(-1).clone()
                    w_backup_parts.append(w_layer)

                    if full_idx in quant_param_indices and QuantizationType == "center":
                        v_layer = v_list[quant_state]
                        q_idx_layer, q_layer = _quantize_with_deadzone(w_layer, v_layer)

                        s_layer = q_layer.clone()
                        s_layer[s_layer.abs() <= sparsity_threshold] = 0.0

                        q_idx_layers.append(q_idx_layer)
                        q_vals_layers.append(q_layer)

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

            if local_rank == 0:
                with torch.no_grad():
                    entropy_total = 0.0
                    q_bytes_chunks = []
                    nz_idx_bytes_chunks = []
                    nz_masks = []

                    total_n = 0
                    total_nz = 0

                    for q_idx_layer, q_vals_layer in zip(q_idx_layers, q_vals_layers):
                        counts = torch.bincount(q_idx_layer, minlength=C).to(torch.float32)
                        probs = counts / counts.sum()
                        probs = probs[probs > 0]

                        entropy_total += float((-(probs * torch.log2(probs)).sum() * counts.sum()).item())

                        q_bytes_chunks.append(_pack_quant_indices(q_idx_layer, C))

                        nz_mask_layer = q_vals_layer.abs() > sparsity_threshold
                        nz_masks.append(nz_mask_layer)

                        total_n += q_idx_layer.numel()
                        total_nz += int(nz_mask_layer.sum().item())

                        nz_idx_bytes_chunks.append(
                            _pack_quant_indices(q_idx_layer[nz_mask_layer], C)
                        )

                    quantized_entropy = round(entropy_total) + 1
                    entropies.append(quantized_entropy)

                    index_bits = _index_bits_for_C(C)

                    # Per-layer metadata:
                    # for each tensor, store two fp32 values, e.g. w0/r or lo/hi.
                    metadata_bits = 32 + 32 + num_param_tensors * 2 * 32

                    q_bytes = b"".join(q_bytes_chunks)
                    zstd_compressed = compress_zstd(q_bytes, level=22)

                    original_bits = total_n * 32
                    compressed_bits = len(zstd_compressed) * 8 + metadata_bits
                    zstd_ratio = compressed_bits / original_bits

                    # Sparse proxy:
                    # global bitmask + compressed non-zero quantized indices.
                    global_nz_mask = torch.cat([m.reshape(-1) for m in nz_masks])
                    mask_np = global_nz_mask.to(torch.uint8).cpu().numpy()

                    bitmask_bytes = pack_bitmaskGPU(mask_np)
                    nz_idx_bytes = b"".join(nz_idx_bytes_chunks)

                    compressed_mask = compress_zstd(bitmask_bytes, level=22)
                    compressed_values = compress_zstd(nz_idx_bytes, level=22)

                    compressed_sparse_bits = (
                        len(compressed_mask) * 8 +
                        len(compressed_values) * 8 +
                        metadata_bits
                    )

                    sparse_ratio = compressed_sparse_bits / original_bits
                    sparsity = 1.0 - float(total_nz) / float(total_n)

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
                    qvals = _percentiles_large_tensor(w_stats, [0.001, 0.01, 0.5, 0.99, 0.999])

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
                print(f"current lr = {optimizer.param_groups[0]['lr']}", flush=True)
                print(f"training_time = {training_time_global}s", flush=True)
                print(f"accuracies = {accuracies}", flush=True)
                print(f"zstd_ratios = {zstd_ratios}", flush=True)
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
        print(f"accuracies = {accuracies}", flush=True)
        print(f"zstd_ratios = {zstd_ratios}", flush=True)
    return
