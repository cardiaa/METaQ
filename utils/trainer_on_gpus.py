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
from utils.quantize_and_compress import compute_entropy, quantize_weights_center, compute_entropyGPU, quantize_weights_centerGPU
from utils.optimization import FISTA, FISTA_leonardo, ProximalBM, test_accuracy, test_accuracyGPU
from utils.weight_utils import initialize_weights
from utils.quantize_and_compress import compress_zstd, BestQuantization, pack_bitmask, pack_bitmaskGPU
from datetime import datetime, timedelta

def train_and_evaluate(model, model_name, criterion, C, lr, lambda_reg, alpha, T1_explicit, T2_explicit, subgradient_step, w0, r, 
                       first_best_indices, BestQuantization_target_acc, final_target_acc, target_zstd_ratio, min_xi, max_xi, upper_c, 
                       lower_c, c1, c2, zeta, l, n_epochs, max_iterations, device, train_optimizer, entropy_optimizer, trainloader,
                        testloader, train_sampler, delta, pruning, QuantizationType, sparsity_threshold, accuracy_tollerance):
    
    #T1_explicit = lambda_reg * alpha
    #T2_explicit = lambda_reg * (1 - alpha)

    local_rank = dist.get_rank() if dist.is_initialized() else 0

    # --- Make sure CUDA device is set consistently on each rank ---
    if device.type == "cuda":
        torch.cuda.set_device(device.index if device.index is not None else local_rank)

    # Selection of the optimizer based on the chosen type.
    if train_optimizer == 'ADAM':
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=T1_explicit)
    elif train_optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=T1_explicit)
    else:
        raise ValueError(f"Unsupported optimizer: {train_optimizer}")

    # Weights Initialization
    min_w, max_w = w0 - r, w0 + r
    v = torch.linspace(min_w, max_w - (max_w - min_w)/C, steps=C, device=device)
    #initialize_weights(model, min_w, max_w)
    with torch.no_grad():
        w = torch.cat([param.detach().view(-1) for param in model.parameters()]).to(device)

    xi = min_xi + (max_xi - min_xi) * torch.rand(C, device=device)
    xi = torch.sort(xi)[0]   
    entropy, accuracy = 0, 0
    accuracies, entropies = [], []

    log = ""
    #delta_regime = delta
    T1_regime = T1_explicit
    T2_regime = T2_explicit
    
    # Training loop
    for epoch in range(n_epochs):
        # Use a schedule to ease training
        if epoch >= 0 and epoch <= 3:
            T1_explicit = T1_regime / 8
            T2_explicit = T2_regime / 8
        elif epoch >= 4 and epoch <= 7:
            T1_explicit = T1_regime / 4
            T2_explicit = T2_regime / 4
        elif epoch >= 8 and epoch <= 11:
            T1_explicit = T1_regime / 2
            T2_explicit = T2_regime / 2
        else:
            T1_explicit = T1_regime
            T2_explicit = T2_regime
        for param_group in optimizer.param_groups:
            param_group['weight_decay'] = T1_explicit   

        # Ensure deterministic sharding for distributed samplers across epochs
        if(train_sampler is not None):
            if hasattr(train_sampler, "set_epoch"):
                train_sampler.set_epoch(epoch)
        #if local_rank == 0:
        #    print(f"Beginning epoch {epoch} at {(datetime.now() + timedelta(hours=2)).strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
        start_time = time.time()
        start_time2 = time.time()
        for i, data in enumerate(trainloader, 0):
            #if i % 100 == 0:
            #if((model_name[:7] == "LeNet-5" or model_name == "LeNet300_100") and delta == 5): 
            #    print(f"Batch {i} of epoch {epoch + 1}: time {round(time.time() - start_time2, 2)}s", flush=True)
            if((model_name == "AlexNet" or model_name == "VGG16") and local_rank == 0):
                if i % 10 == 0:
                    print(f"Batch {i} of epoch {epoch + 1}: time {round(time.time() - start_time2, 2)}s", flush=True)
                    start_time2 = time.time()
                    """
                    w = torch.cat([param.detach().view(-1) for param in model.parameters()]).to(device)
                    num_samples = 1000000
                    idx = torch.randperm(w.numel(), device=w.device)[:num_samples]
                    w_sample = w[idx]                
                    qs = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
                    qs = qs.to(w.device)
                    valori = torch.quantile(w_sample, qs)
                    valori_rounded = [round(v.item(), 4) for v in valori]
                    print(f"Quartiles of weights: {valori_rounded}", flush=True)
                    """
                    
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backpropagation
            loss.backward()

            grads_loss = [p.grad.detach().clone() if p.grad is not None else None
              for p in model.parameters()]
            
            if(alpha != 1):
                with torch.no_grad():
                    w = torch.cat([param.detach().view(-1) for param in model.parameters()]).to(device)

                    #unique_weights = torch.unique(w).numel() # Alternative version
                    #indices = torch.searchsorted(v, w, right=True) - 1
                    #indices = torch.clamp(indices, min=0)
                    #w_quantized = v[indices]

                    zeta *= 1 + l
                    l = l / 1.5

                with torch.no_grad():
                    if(entropy_optimizer == 'FISTA'):
                        #xi, beta_tensor = FISTA(xi, v, w_quantized, C, upper_c, lower_c, delta, 
                        #                        subgradient_step, device, max_iterations, pruning) # Alternative version
                        
                        #xi, beta_tensor = torch.zeros(C, dtype=torch.int32), torch.zeros(len(w), dtype=torch.int32)
                        #xi = xi.to(device)
                        #beta_tensor = beta_tensor.to(device)
                        
                        """
                        xi, beta_tensor = FISTA(xi, v, w, C, upper_c, lower_c, delta, 
                                                subgradient_step, device, max_iterations, pruning) 
                        """
                        xi, beta_tensor = FISTA_leonardo(xi, v, w, C, upper_c, lower_c, delta, 
                                                        subgradient_step, device, max_iterations, pruning)
                        
                    elif(entropy_optimizer == 'PROXIMAL BM'):
                        #xi, beta_tensor = ProximalBM(xi, v, w_quantized, C, upper_c, lower_c, delta, 
                        #                             zeta, subgradient_step, device, max_iterations, pruning) # Alternative version
                        xi, beta_tensor = ProximalBM(xi, v, w, C, upper_c, lower_c, delta, 
                                                    zeta, subgradient_step, device, max_iterations, pruning)       

                # Update of ∇ɸ
                with torch.no_grad():
                    idx = 0
                    for param in model.parameters():
                        numel = param.numel()
                        if param.grad is not None:
                            update = (T2_explicit * (-beta_tensor[idx:idx + numel])).view(param.size())
                            param.grad.add_(update)
                        idx += numel
                
                if(i == 310):
                    # Debug: compute and print norms of gradients
                    with torch.no_grad():
                        idx = 0
                        norm_l2_total = 0.0
                        norm_custom_total = 0.0

                        for param in model.parameters():
                            numel = param.numel()
                            if param.grad is not None:

                                # ---- gradiente L2 "grezzo" ----
                                grad_l2_core = param.detach()  # SOLO w
                                norm_l2_total += torch.norm(grad_l2_core).item()

                                # ---- gradiente custom "grezzo" ----
                                grad_custom_core = (-beta_tensor[idx:idx + numel]).view(param.size())
                                norm_custom_total += torch.norm(grad_custom_core).item()

                            idx += numel

                        # Stampa delle norme
                        norm_loss_grad = 0.0
                        for g in grads_loss:
                            if g is not None:
                                norm_loss_grad += torch.norm(g).item()

                        if local_rank == 0:
                            print("--- Gradient Norms at Batch 310 ---", flush=True)
                            print(f"L2 grad norm (core): {norm_l2_total:.4f}\n"
                                f"Custom grad norm (core): {norm_custom_total:.4f}\n"
                                f"Loss grad norm (pure): {norm_loss_grad:.4f}\n"
                                f"Weighted L2 grad norm: {(norm_l2_total*T1_explicit):.4f}\n"
                                f"Weighted Custom grad norm: {(norm_custom_total*T2_explicit):.4f}",
                                flush=True)
                            print("-------------------------------------", flush=True)

            optimizer.step()

        training_time = round(time.time() - start_time)
        #if local_rank == 0:
        #    print(f"Epoch {epoch + 1}: training_time = {training_time}s\n", flush=True)
        if(model_name[:7] == "LeNet-5" and delta == 5): # To modify if delta's tests are different
            print(f"Epoch {epoch + 1}: training_time = {training_time}s\n", flush=True)
        if(model_name == "LeNet300_100" and delta == 5): # To modify if delta's tests are different
            print(f"Epoch {epoch + 1}: training_time = {training_time}s\n", flush=True)            

        # --- Metrics & Logging ---
        if (epoch % 1 == 0 or epoch == n_epochs - 1):

            # --- 0) Synchronize all ranks BEFORE evaluation/CPU-heavy work ---
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            if dist.is_initialized():
                if dist.get_backend() == "nccl" and device.type == "cuda":
                    dist.barrier(device_ids=[torch.cuda.current_device()])
                else:
                    dist.barrier()

            # --- 1) Compute non-quantized accuracy on ALL ranks ---
            with torch.no_grad():
                accuracy = test_accuracyGPU(model, testloader, device)  # all ranks must participate
            if local_rank == 0:
                accuracies.append(accuracy)

            # --- 1b) Barrier: ensure all ranks finished accuracy ---
            if dist.is_initialized():
                if dist.get_backend() == "nccl" and device.type == "cuda":
                    dist.barrier(device_ids=[torch.cuda.current_device()])
                else:
                    dist.barrier()

            # --- 2) Quantized/Sparse evaluation WITHOUT deepcopy(model) ---
            # Helper: load a flat tensor into model parameters (in-place).
            def _load_flat_params_(flat: torch.Tensor):
                offset = 0
                for p_ in model.parameters():
                    n = p_.numel()
                    p_.data.copy_(flat[offset:offset + n].view_as(p_))
                    offset += n

            # 2.0) Backup current weights on EVERY rank (fast restore)
            with torch.no_grad():
                w_backup = torch.cat([p_.detach().view(-1) for p_ in model.parameters()]).clone()

            # 2.1) Rank 0 computes CPU-heavy stuff (entropy/quantization/compression) using CPU copy
            if local_rank == 0:
                with torch.no_grad():
                    w_cpu = w_backup.detach().cpu()

                # --- Non-quantized entropy ---
                w_np = w_cpu.numpy().astype(np.float32)
                entropy = round(compute_entropyGPU(w_np.tolist())) + 1
                entropies.append(entropy)

                # --- Quantization on CPU ---
                if QuantizationType == "center":
                    v_centers_cpu = ((v[:-1] + v[1:]) / 2).detach().cpu()
                    wq_cpu = quantize_weights_centerGPU(w_cpu, v.detach().cpu(), v_centers_cpu, device="cpu")
                else:
                    wq_cpu = w_cpu.clone()

                # Normalize -0.0 to +0.0 (stable entropy/compression)
                wq_np = wq_cpu.numpy().astype(np.float32)
                mask_negzero = np.signbit(wq_np) & (wq_np == 0.0)
                wq_np[mask_negzero] = 0.0

                # --- Quantized entropy ---
                quantized_entropy = round(compute_entropyGPU(wq_np.tolist())) + 1

                # --- Bytes and compression (quantized) ---
                input_bytes = wq_np.tobytes()
                zstd_compressed = compress_zstd(input_bytes, level=22)
                original_size_bytes = len(input_bytes)
                zstd_ratio = len(zstd_compressed) / original_size_bytes

                # --- Sparse representation (from quantized weights) ---
                mask = (np.abs(wq_np) > sparsity_threshold).astype(np.uint8)
                nonzero_values = wq_np[mask == 1]
                bitmask_bytes = pack_bitmaskGPU(mask.tolist())
                packed_nonzeros = nonzero_values.tobytes()
                compressed_mask = compress_zstd(bitmask_bytes, level=22)
                compressed_values = compress_zstd(packed_nonzeros, level=22)
                sparse_ratio = (len(compressed_mask) + len(compressed_values)) / original_size_bytes
                sparsity = 1.0 - mask.sum() / mask.size

                # Build sparse weights (still on CPU)
                ws_cpu = torch.from_numpy(wq_np.copy())
                ws_cpu[mask == 0] = 0.0
            else:
                entropy = None
                quantized_entropy = None
                zstd_ratio = None
                sparse_ratio = None
                sparsity = None
                wq_cpu = None
                ws_cpu = None

            # 2.2) Broadcast QUANTIZED flat weights to all ranks
            flat_q = torch.empty_like(w_backup)
            if local_rank == 0:
                flat_q.copy_(wq_cpu.to(device))
            if dist.is_initialized():
                dist.broadcast(flat_q, src=0)

            # 2.3) Evaluate quantized accuracy on ALL ranks (all ranks participate)
            with torch.no_grad():
                _load_flat_params_(flat_q)
                quantized_accuracy = test_accuracyGPU(model, testloader, device)

            # 2.4) Broadcast SPARSE flat weights to all ranks
            flat_s = torch.empty_like(w_backup)
            if local_rank == 0:
                flat_s.copy_(ws_cpu.to(device))
            if dist.is_initialized():
                dist.broadcast(flat_s, src=0)

            # 2.5) Evaluate sparse accuracy on ALL ranks
            with torch.no_grad():
                _load_flat_params_(flat_s)
                sparse_accuracy = test_accuracyGPU(model, testloader, device)

            # 2.6) Restore original weights on ALL ranks
            with torch.no_grad():
                _load_flat_params_(w_backup)

            # --- 2.7) Logging rank 0 ---
            if local_rank == 0:
                training_time = round(time.time() - start_time)
                if epoch == 0:
                    log += f"delta = {delta}\n"
                log += (
                    f"Epoch {epoch + 1}: "
                    f"A_NQ = {accuracy}, H_NQ = {entropy}, "
                    f"A_Q = {quantized_accuracy}, H_Q = {quantized_entropy}, "
                    f"zstd_ratio = {zstd_ratio:.2%}, sparse_ratio = {sparse_ratio:.2%}, "
                    f"sparsity = {sparsity:.2%} , sparse_accuracy = {sparse_accuracy}, training_time = {training_time}s\n\n"
                )
                if model_name in ("AlexNet", "VGG16"):
                    print(
                        f"Epoch {epoch + 1}: "
                        f"A_NQ = {accuracy}, H_NQ = {entropy}, "
                        f"A_Q = {quantized_accuracy}, H_Q = {quantized_entropy}, "
                        f"zstd_ratio = {zstd_ratio:.2%}, sparse_ratio = {sparse_ratio:.2%}, "
                        f"sparsity = {sparsity:.2%} , sparse_accuracy = {sparse_accuracy}, training_time = {training_time}s\n",
                        flush=True
                    )

            # --- 3) Final barrier: allow all ranks to resume training ---
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            if dist.is_initialized():
                if dist.get_backend() == "nccl" and device.type == "cuda":
                    dist.barrier(device_ids=[torch.cuda.current_device()])
                else:
                    dist.barrier()


        """
        # Saving a better model
        if(accuracies[-1] >= BestQuantization_target_acc):
            log = BestQuantization(log=log, C=C, r=r, delta=delta, epoch=epoch, min_w=min_w, max_w=max_w, w=w, c1=c1, c2=c2,
                                   final_target_acc=final_target_acc, target_zstd_ratio=target_zstd_ratio,
                                   sparsity_threshold=sparsity_threshold, QuantizationType=QuantizationType, model=model, 
                                   testloader=testloader, accuracy=accuracy, device=device, first_best_indices=first_best_indices, 
                                   accuracy_tollerance=accuracy_tollerance)
            BestQuantization_target_acc = accuracies[-1] 
        """

        # ---------------------------------------------------------------------------------------------------------
        
        """
        # No-pruning exit conditions
        if(pruning == "N"):
            # Entropy exit conditions
            # After the tenth epoch I must have entropy below 600000
            if(epoch >= 10 and entropies[-1] >= 600000):
                log += (
                    f"Entropy is not decreasing enough! (E1.1), delta: {delta}\n"
                )
                log += "-"*60

                print(log, flush = True)
                return
            
            # After the 30th epoch I must not have entropy above 200000 for 4 epochs in a row
            if(epoch >= 30):
                if(entropies[-1] >= 200000 and entropies[-2] >= 200000 and entropies[-3] >= 200000 and entropies[-4] >= 200000):
                    log += (
                        f"Entropy is not decreasing enough! (E2.1), delta: {delta}\n"
                    )
                    log += "-"*60

                    print(log, flush = True)
                    return
                
            # ---------------------------------------------------------------------------------------------------------
            # Accuracy exit condition
            # After the first epoch I must have accuracy above 60%
            if(epoch >= 1 and accuracies[-1] <= 60):
                log += (
                    f"Accuracy is too low! (A1.1), delta: {delta}\n"
                )
                log += "-"*60

                print(log, flush = True)
                return
                
            # After the 20th epoch I must have accuracy above 96%
            if(epoch >= 20 and accuracies[-1] <= 96):
                log += (
                    f"Accuracy is too low! (A1.2), delta: {delta}\n"
                )
                log += "-"*60

                print(log, flush = True)
                return

            # After the 80th epoch I must have accuracy above 98%
            if(epoch >= 80 and accuracies[-1] <= 98):
                log += (
                    f"Accuracy is too low! (A1.3), delta: {delta}\n"
                )
                log += "-"*60

                print(log, flush = True)
                return
            
            # After the 30th epoch I must not have accuracy below 90% for 4 epochs in a row
            if(epoch >= 30):
                if(accuracies[-1] <= 90 and accuracies[-2] <= 90 and accuracies[-3] <= 90 and accuracies[-4] <= 90):
                    log += (
                        f"Accuracy is too low! (A2.1), delta: {delta}\n"
                    )
                    log += "-"*60

                    print(log, flush = True)
                    return  
            
            # ... ADD OTHER EXIT CONDITIONS IF NECESSARY...   
        # ---------------------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------------------------------
        # Pruning exit conditions
        elif(pruning == "Y"):
            # Entropy exit conditions
            # After the tenth epoch I must have entropy below 200000
            
            if(epoch >= 0 and quantized_entropy >= 400000):
                log += (
                    f"Entropy is not decreasing enough! (E1.1), delta: {delta}\n"
                )
                log += "-"*60

                print(log, flush = True)
                return
            
            # ---------------------------------------------------------------------------------------------------------
            # Accuracy exit condition
            # After the first epoch I must have accuracy above 30%
            if(epoch >= 0 and accuracies[-1] <= 30):
                log += (
                    f"Accuracy is too low! (A1.1), delta: {delta}\n"
                )
                log += "-"*60

                print(log, flush = True)
                return
            
            # After the 10th epoch I must have accuracy above 94%
            if(epoch >= 9 and accuracies[-1] <= 94):
                log += (
                    f"Accuracy is too low! (A1.2), delta: {delta}\n"
                )
                log += "-"*60

                print(log, flush = True, end = "")
                return
            """
            
            # ... ADD OTHER EXIT CONDITIONS IF NECESSARY...   
           
        # ---------------------------------------------------------------------------------------------------------
        
        gc.collect()
        torch.cuda.empty_cache()
    
    log += "-"*60
    if local_rank == 0:
        print(log, flush = True)
    return