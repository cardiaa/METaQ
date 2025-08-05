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
from utils.quantize_and_compress import compute_entropy, quantize_weights_center
from utils.optimization import FISTA, ProximalBM, test_accuracy
from utils.weight_utils import initialize_weights
from utils.quantize_and_compress import compress_zstd, BestQuantization, pack_bitmask
from datetime import datetime, timedelta

def train_and_evaluate(model, model_name, criterion, C, lr, lambda_reg, alpha, subgradient_step, w0, r, first_best_indices,
                        BestQuantization_target_acc, final_target_acc, target_zstd_ratio, min_xi, max_xi, upper_c, lower_c, 
                        c1, c2, zeta, l, n_epochs, max_iterations, device, train_optimizer, entropy_optimizer, trainloader,
                        testloader, train_sampler, delta, pruning, QuantizationType, sparsity_threshold, accuracy_tollerance):

    local_rank = dist.get_rank() if dist.is_initialized() else 0

    # Selection of the optimizer based on the chosen type.
    if train_optimizer == 'ADAM':
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=lambda_reg * alpha)
    elif train_optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=lambda_reg * alpha)

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
        train_sampler.set_epoch(epoch)
        if local_rank == 0:
            print(f"Beginning epoch {epoch} at {(datetime.now() + timedelta(hours=2)).strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
        start_time = time.time()
        start_time2 = time.time()
        for i, data in enumerate(trainloader, 0):
            #if i % 100 == 0:
            if local_rank == 0:
                print(f"Batch {i} of epoch {epoch + 1} [GPU {local_rank}]: time {round(time.time() - start_time2, 2)}s", flush=True)
            start_time2 = time.time()
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backpropagation
            loss.backward()
            
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
        if local_rank == 0:
            print(f"Epoch {epoch + 1}: training_time = {training_time}s\n", flush=True)
        """
        t0 = time.time()
        with torch.no_grad():
            w = torch.cat([param.detach().view(-1) for param in model.parameters()]).to(device)
        if local_rank == 0:
            print("Debug 1 - Time:", round(time.time() - t0, 2), "s", flush=True)
        t0 = time.time()
        with torch.no_grad():
            accuracy = test_accuracy(model, testloader, device)
        accuracies.append(accuracy)
        if local_rank == 0:
            print("Debug 2 - Time:", round(time.time() - t0, 2), "s", flush=True)
        t0 = time.time()
        entropy = round(compute_entropy(w.tolist())) + 1
        entropies.append(entropy)
        if local_rank == 0:
            print("Debug 3 - Time:", round(time.time() - t0, 2), "s", flush=True)
        t0 = time.time()
        if(QuantizationType == "center"): # Quantize weights using central values
            v_centers = (v[:-1] + v[1:]) / 2
            v_centers = torch.cat([v_centers, v[-1:]]) # Add final value to handle the last bucket
            w_quantized = quantize_weights_center(w, v, v_centers, device)
        if local_rank == 0:
            print("Debug 4 - Time:", round(time.time() - t0, 2), "s", flush=True)
        t0 = time.time()
        model_quantized = copy.deepcopy(model).to(device)
        start_idx = 0
        for param in model_quantized.parameters():
            numel = param.data.numel()
            param.data.copy_(w_quantized[start_idx:start_idx + numel].view(param.data.size()))
            start_idx += numel
        with torch.no_grad():
            model_quantized.eval()
        quantized_accuracy = test_accuracy(model_quantized, testloader, device)
        if local_rank == 0:
            print("Debug 5 - Time:", round(time.time() - t0, 2), "s", flush=True)
        t0 = time.time()
        with torch.no_grad():
            encoded_list = [float(elem) if float(elem) != -0.0 else 0.0 for elem in w_quantized]
        quantized_entropy = round(compute_entropy(encoded_list)) + 1
        input_bytes = b''.join(struct.pack('f', num) for num in encoded_list)
        zstd_compressed = compress_zstd(input_bytes, level=3)
        original_size_bytes = len(input_bytes)
        zstd_size = len(zstd_compressed)
        zstd_ratio = zstd_size / original_size_bytes  
        if local_rank == 0:
            print("Debug 6 - Time:", round(time.time() - t0, 2), "s", flush=True)
        t0 = time.time()
        # --- Sparse compression ---
        mask = [1 if abs(val) > sparsity_threshold else 0 for val in encoded_list]
        nonzero_values = [val for val in encoded_list if abs(val) > sparsity_threshold]
        bitmask_bytes = pack_bitmask(mask)
        packed_nonzeros = b''.join(struct.pack('f', val) for val in nonzero_values)
        compressed_mask = compress_zstd(bitmask_bytes, level=3)
        compressed_values = compress_zstd(packed_nonzeros, level=3)
        sparse_compressed_size = len(compressed_mask) + len(compressed_values)
        sparse_ratio = sparse_compressed_size / original_size_bytes
        sparsity = 1.0 - sum(mask) / len(mask) 
        if local_rank == 0:
            print("Debug 7 - Time:", round(time.time() - t0, 2), "s", flush=True)
        t0 = time.time()
        # Applies the sparsity mask to quantized weights
        w_sparse = torch.as_tensor(encoded_list, device=device)
        sparse_mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=device)
        w_sparse[~sparse_mask_tensor] = 0.0
        if local_rank == 0:
            print("Debug 8 - Time:", round(time.time() - t0, 2), "s", flush=True)
        t0 = time.time()
        # Build a new sparsified model
        model_sparse = copy.deepcopy(model).to(device)
        start_idx = 0
        for param in model_sparse.parameters():
            numel = param.data.numel()
            param.data.copy_(w_sparse[start_idx:start_idx + numel].view(param.data.size()))
            start_idx += numel
        with torch.no_grad():
            model_sparse.eval()
        if local_rank == 0:
            print("Debug 9 - Time:", round(time.time() - t0, 2), "s", flush=True)
        # Evaluate the accuracy of the sparsified model
        with torch.no_grad():
            sparse_accuracy = test_accuracy(model_sparse, testloader, device)

        training_time = round(time.time() - start_time)

        if(epoch == 0):
            log += f"delta = {delta}\n"
    
        log += (
            f"Epoch {epoch + 1}: "
            f"A_NQ = {accuracy}, H_NQ = {entropy}, "
            f"A_Q = {quantized_accuracy}, H_Q = {quantized_entropy}, "
            f"zstd_ratio = {zstd_ratio:.2%}, sparse_ratio = {sparse_ratio:.2%}, "
            f"sparsity = {sparsity:.2%} , sparse_accuracy = {sparse_accuracy}, training_time = {training_time}s\n"     
        )

        if local_rank == 0:
            print(
                f"Epoch {epoch + 1}: "
                f"A_NQ = {accuracy}, H_NQ = {entropy}, "
                f"A_Q = {quantized_accuracy}, H_Q = {quantized_entropy}, "
                f"zstd_ratio = {zstd_ratio:.2%}, sparse_ratio = {sparse_ratio:.2%}, "
                f"sparsity = {sparsity:.2%} , sparse_accuracy = {sparse_accuracy}, training_time = {training_time}s\n", 
                flush=True              
            )

        # Saving a better model
        if(accuracies[-1] >= BestQuantization_target_acc):
            log = BestQuantization(log=log, C=C, r=r, delta=delta, epoch=epoch, min_w=min_w, max_w=max_w, w=w, c1=c1, c2=c2,
                                   final_target_acc=final_target_acc, target_zstd_ratio=target_zstd_ratio,
                                   sparsity_threshold=sparsity_threshold, QuantizationType=QuantizationType, model=model, 
                                   testloader=testloader, accuracy=accuracy, device=device, first_best_indices=first_best_indices, 
                                   accuracy_tollerance=accuracy_tollerance)
            BestQuantization_target_acc = accuracies[-1] 
        
        # ---------------------------------------------------------------------------------------------------------
        """
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
