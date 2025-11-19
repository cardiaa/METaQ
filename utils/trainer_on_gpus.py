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
from utils.optimization import FISTA, ProximalBM, test_accuracy, test_accuracyGPU
from utils.weight_utils import initialize_weights
from utils.quantize_and_compress import compress_zstd, BestQuantization, pack_bitmask, pack_bitmaskGPU
from datetime import datetime, timedelta

def train_and_evaluate(model, model_name, criterion, C, lr, lambda_reg, alpha, subgradient_step, w0, r, first_best_indices,
                        BestQuantization_target_acc, final_target_acc, target_zstd_ratio, min_xi, max_xi, upper_c, lower_c, 
                        c1, c2, zeta, l, n_epochs, max_iterations, device, train_optimizer, entropy_optimizer, trainloader,
                        testloader, train_sampler, delta, pruning, QuantizationType, sparsity_threshold, accuracy_tollerance):

    local_rank = dist.get_rank() if dist.is_initialized() else 0

    # --- Make sure CUDA device is set consistently on each rank ---
    if device.type == "cuda":
        torch.cuda.set_device(device.index if device.index is not None else local_rank)

    # Selection of the optimizer based on the chosen type.
    if train_optimizer == 'ADAM':
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=lambda_reg * alpha)
    elif train_optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=lambda_reg * alpha)
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

    # Training loop
    for epoch in range(n_epochs):
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
                    w = torch.cat([param.detach().view(-1) for param in model.parameters()]).to(device)
                    num_samples = 1000000
                    idx = torch.randperm(w.numel(), device=w.device)[:num_samples]
                    w_sample = w[idx]                
                    qs = torch.tensor([0.0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 0.99, 0.999, 0.9999, 0.99999, 1.0])
                    qs = qs.to(w.device)
                    valori = torch.quantile(w_sample, qs)
                    valori_rounded = [round(v.item(), 4) for v in valori]
                    print("Quantiles of weights:", flush=True)
                    print([f"{q:.5f}" for q in qs.tolist()], flush=True)
                    print(f"{valori_rounded}", flush=True)
                
            start_time2 = time.time()
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backpropagation
            loss.backward()
            
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
                        
                        xi, beta_tensor = FISTA(xi, v, w, C, upper_c, lower_c, delta, 
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
                            update = ((1 - alpha) * lambda_reg * (-beta_tensor[idx:idx + numel])).view(param.size())
                            param.grad.add_(update)
                        idx += numel
            
            optimizer.step()

        training_time = round(time.time() - start_time)
        #if local_rank == 0:
        #    print(f"Epoch {epoch + 1}: training_time = {training_time}s\n", flush=True)
        if(model_name[:7] == "LeNet-5" and delta == 5): # To modify if delta's tests are different
            print(f"Epoch {epoch + 1}: training_time = {training_time}s\n", flush=True)
        if(model_name == "LeNet300_100" and delta == 5): # To modify if delta's tests are different
            print(f"Epoch {epoch + 1}: training_time = {training_time}s\n", flush=True)            

        # --- Metrics & Logging ---
        if epoch % 1 == 0 or epoch == n_epochs - 1:

            # --- 0) Synchronize all ranks BEFORE evaluation/CPU-heavy work ---
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            if dist.is_initialized():
                if dist.get_backend() == "nccl" and device.type == "cuda":
                    dist.barrier(device_ids=[torch.cuda.current_device()])
                else:
                    dist.barrier()

            # --- 1) Compute accuracy on ALL ranks ---
            t0_acc = time.time()
            with torch.no_grad():
                accuracy = test_accuracyGPU(model, testloader, device)  # all ranks must participate
            if local_rank == 0:
                accuracies.append(accuracy)
                #print("Debug 1 - Time:", round(time.time() - t0_acc, 2), "s", flush=True)

            # --- Optional sync: make sure all ranks finished accuracy ---
            if dist.is_initialized():
                if dist.get_backend() == "nccl" and device.type == "cuda":
                    dist.barrier(device_ids=[torch.cuda.current_device()])
                else:
                    dist.barrier()

            # --- 2) Rank 0 performs CPU-heavy computations ---
            if local_rank == 0:
                # --- 2.1) Collect weights on CPU ---
                t0 = time.time()
                with torch.no_grad():
                    w = torch.cat([param.detach().view(-1) for param in model.parameters()]).cpu()
                #print("Debug 2 - Time:", round(time.time() - t0, 2), "s", flush=True)

                # --- 2.2) Non-quantized entropy ---
                t0 = time.time()
                w_np = w.numpy().astype(np.float32)
                entropy = round(compute_entropyGPU(w_np.tolist())) + 1
                entropies.append(entropy)
                #print("Debug 3 - Time:", round(time.time() - t0, 2), "s", flush=True)

                # --- 2.3) Quantization on CPU ---
                t0 = time.time()
                if QuantizationType == "center":
                    v_centers_cpu = ((v[:-1] + v[1:]) / 2).cpu()
                    w_quantized = quantize_weights_centerGPU(w, v.cpu(), v_centers_cpu, device='cpu')
                else:
                    w_quantized = w.clone()
                #print("Debug 4 - Time:", round(time.time() - t0, 2), "s", flush=True)

                # --- 2.4) Build CPU model with quantized weights ---
                t0 = time.time()
                model_quantized = copy.deepcopy(model).cpu()
                start_idx = 0
                wq_np = w_quantized.numpy().astype(np.float32)
                for param in model_quantized.parameters():
                    numel = param.data.numel()
                    param.data.copy_(torch.from_numpy(wq_np[start_idx:start_idx + numel].reshape(param.data.size())))
                    start_idx += numel
                model_quantized.eval()
            else:
                # Gli altri rank devono avere la variabile pronta per il collective,
                # ma il contenuto può essere un dummy model
                model_quantized = copy.deepcopy(model).cpu()
                model_quantized.eval()

            # --- 2.5) Evaluate quantized model accuracy on ALL ranks ---
            model_quantized = model_quantized.to(device)  # tutti i rank spostano il model sul device
            t0_qacc = time.time()
            with torch.no_grad():
                quantized_accuracy = test_accuracyGPU(model_quantized, testloader, device)
            #if local_rank == 0:
                #print("Debug 5 - Time:", round(time.time() - t0_qacc, 2), "s", flush=True)

            if local_rank == 0:
                # --- 2.6) Normalize -0.0 to +0.0 ---
                t0 = time.time()
                arr = wq_np
                mask_negzero = np.signbit(arr) & (arr == 0.0)
                arr[mask_negzero] = 0.0
                #print("Debug 6 - Time:", round(time.time() - t0, 2), "s", flush=True)

                # --- 2.7) Quantized entropy ---
                t0 = time.time()
                quantized_entropy = round(compute_entropyGPU(arr.tolist())) + 1
                #print("Debug 7 - Time:", round(time.time() - t0, 2), "s", flush=True)

                # --- 2.8) Bytes and compression ---
                t0 = time.time()
                input_bytes = arr.tobytes()
                zstd_compressed = compress_zstd(input_bytes, level=22)
                original_size_bytes = len(input_bytes)
                zstd_size = len(zstd_compressed)
                zstd_ratio = zstd_size / original_size_bytes
                #print("Debug 8 - Time:", round(time.time() - t0, 2), "s", flush=True)

                # --- 2.9) Sparse representation ---
                t0 = time.time()
                mask = (np.abs(arr) > sparsity_threshold).astype(np.uint8)
                nonzero_values = arr[mask == 1]
                bitmask_bytes = pack_bitmaskGPU(mask.tolist())
                packed_nonzeros = nonzero_values.tobytes()
                compressed_mask = compress_zstd(bitmask_bytes, level=22)
                compressed_values = compress_zstd(packed_nonzeros, level=22)
                sparse_compressed_size = len(compressed_mask) + len(compressed_values)
                sparse_ratio = sparse_compressed_size / original_size_bytes
                sparsity = 1.0 - mask.sum() / mask.size
                #print("Debug 9 - Time:", round(time.time() - t0, 2), "s", flush=True)

                # --- 2.10) Build sparse model and evaluate accuracy ---
                t0 = time.time()
                w_sparse_np = arr.copy()
                w_sparse_np[mask == 0] = 0.0
                model_sparse = copy.deepcopy(model).cpu()
                start_idx = 0
                for param in model_sparse.parameters():
                    numel = param.data.numel()
                    param.data.copy_(torch.from_numpy(w_sparse_np[start_idx:start_idx + numel].reshape(param.data.size())))
                    start_idx += numel
                model_sparse.eval()
            else:
                # Dummy model per gli altri rank
                model_sparse = copy.deepcopy(model).cpu()
                model_sparse.eval()

            # Tutti i rank eseguono la valutazione finale
            model_sparse = model_sparse.to(device)
            t0_sacc = time.time()
            with torch.no_grad():
                sparse_accuracy = test_accuracyGPU(model_sparse, testloader, device)
            #if local_rank == 0:
            #    print("Debug 10 - Time:", round(time.time() - t0_sacc, 2), "s", flush=True)

            # --- 2.11) Logging rank 0 ---
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
                if(model_name == "AlexNet" or model_name == "VGG16"): # In this case I'm not training the model 
                                                                      # a lot of times in parallel with Dantzig or Fenchel
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