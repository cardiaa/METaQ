import torch
import time 
import numpy as np
import copy
import struct
import sys
from torch.optim import Adam, SGD
from utils.quantize_and_compress import compute_entropy, quantize_weights_center
from utils.optimization import FISTA, ProximalBM, test_accuracy
from utils.weight_utils import initialize_weights
from utils.quantize_and_compress import compress_zstd, BestQuantization, pack_bitmask

def train_and_evaluate(model, criterion, C, lr, lambda_reg, alpha, subgradient_step, w0, r, first_best_indices,
                        BestQuantization_target_acc, final_target_acc, target_zstd_ratio, min_xi, max_xi, upper_c, lower_c, 
                        c1, c2, zeta, l, n_epochs, max_iterations, device, train_optimizer, entropy_optimizer, 
                        trainloader, testloader, delta, pruning, QuantizationType, sparsity_threshold, accuracy_tollerance):
    
    torch.set_num_threads(1)
    
    # Selection of the optimizer based on the chosen type.
    if train_optimizer == 'ADAM':
        optimizer = Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=lambda_reg * alpha)
    elif train_optimizer == 'SGD':
        optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=lambda_reg * alpha)
    
    # Weights Initialization
    min_w, max_w = w0 - r, w0 + r
    v = torch.linspace(min_w, max_w - (max_w - min_w)/C, steps=C, device=device)
    initialize_weights(model, min_w, max_w)
    w = torch.cat([param.data.view(-1) for param in model.parameters()]).to(device)
    xi = min_xi + (max_xi - min_xi) * torch.rand(C, device=device)
    xi = torch.sort(xi)[0]   
    entropy, accuracy = 0, 0
    accuracies, entropies = [], []

    log = ""

    # Training loop
    for epoch in range(n_epochs):
        start_time = time.time()

        for i, data in enumerate(trainloader, 0):
            inputs, targets = data
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backpropagation
            loss.backward()
            optimizer.step()

            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            w = torch.cat([param.data.view(-1) for param in model.parameters()])
            #unique_weights = torch.unique(w).numel() # Alternative version
            #indices = torch.searchsorted(v, w, right=True) - 1
            #indices = torch.clamp(indices, min=0)
            #w_quantized = v[indices]

            zeta *= 1 + l
            l = l / 1.5
            if(entropy_optimizer == 'FISTA'):
                #xi, beta_tensor = FISTA(xi, v, w_quantized, C, upper_c, lower_c, delta, 
                #                        subgradient_step, device, max_iterations, pruning) # Alternative version
                xi, beta_tensor = FISTA(xi, v, w, C, upper_c, lower_c, delta, 
                                        subgradient_step, device, max_iterations, pruning) 
            elif(entropy_optimizer == 'PROXIMAL BM'):
                #xi, beta_tensor = ProximalBM(xi, v, w_quantized, C, upper_c, lower_c, delta, 
                #                             zeta, subgradient_step, device, max_iterations, pruning) # Alternative version
                xi, beta_tensor = ProximalBM(xi, v, w, C, upper_c, lower_c, delta, 
                                             zeta, subgradient_step, device, max_iterations, pruning)       

            # Update of ∇ɸ
            idx = 0
            for param in model.parameters():
                numel = param.numel()
                if param.grad is not None:
                    param_grad = param.grad.view(-1)
                else:
                    param_grad = torch.zeros_like(param.data.view(-1)).to(device)
                param_grad += (1 - alpha) * lambda_reg * (- beta_tensor[idx:idx + numel]) #here there is a minus sign in front of the betas multipliers
                param.grad = param_grad.view(param.size())
                idx += numel
            
            loss.backward()
            optimizer.step()
        
        w = torch.cat([param.data.view(-1) for param in model.parameters()]).to(device)
        
        accuracy = test_accuracy(model, testloader, device)
        accuracies.append(accuracy)
        entropy = round(compute_entropy(w.tolist())) + 1
        entropies.append(entropy)

        if(QuantizationType == "center"): # Quantize weights using central values
            v_centers = (v[:-1] + v[1:]) / 2
            v_centers = torch.cat([v_centers, v[-1:]]) # Add final value to handle the last bucket
            w_quantized = quantize_weights_center(w, v, v_centers)
        
        model_quantized = copy.deepcopy(model).to(device)
        start_idx = 0
        for param in model_quantized.parameters():
            numel = param.data.numel()
            param.data = w_quantized[start_idx:start_idx + numel].view(param.data.size())
            start_idx += numel
        model_quantized.eval()
        quantized_accuracy = test_accuracy(model_quantized, testloader, device)

        encoded_list = [float(elem) if float(elem) != -0.0 else 0.0 for elem in w_quantized]
        quantized_entropy = round(compute_entropy(encoded_list)) + 1
        input_bytes = b''.join(struct.pack('f', num) for num in encoded_list)
        zstd_compressed = compress_zstd(input_bytes)
        original_size_bytes = len(input_bytes)
        zstd_size = len(zstd_compressed)
        zstd_ratio = zstd_size / original_size_bytes  

        # --- Sparse compression ---
        mask = [1 if abs(val) > sparsity_threshold else 0 for val in encoded_list]
        nonzero_values = [val for val in encoded_list if abs(val) > sparsity_threshold]
        bitmask_bytes = pack_bitmask(mask)
        packed_nonzeros = b''.join(struct.pack('f', val) for val in nonzero_values)
        compressed_mask = compress_zstd(bitmask_bytes)
        compressed_values = compress_zstd(packed_nonzeros)
        sparse_compressed_size = len(compressed_mask) + len(compressed_values)
        sparse_ratio = sparse_compressed_size / original_size_bytes
        sparsity = 1.0 - sum(mask) / len(mask) 

        # Applies the sparsity mask to quantized weights
        w_sparse = torch.tensor(encoded_list).to(device)
        sparse_mask_tensor = torch.tensor(mask, dtype=torch.bool).to(device)
        w_sparse[~sparse_mask_tensor] = 0.0

        # Build a new sparsified model
        model_sparse = copy.deepcopy(model).to(device)
        start_idx = 0
        for param in model_sparse.parameters():
            numel = param.data.numel()
            param.data = w_sparse[start_idx:start_idx + numel].view(param.data.size())
            start_idx += numel
        model_sparse.eval()

        # Evaluate the accuracy of the sparsified model
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

        # Saving a better model
        if(accuracies[-1] >= BestQuantization_target_acc):
            log = BestQuantization(log=log, C=C, r=r, epoch=epoch, min_w=min_w, max_w=max_w, w=w, c1=c1, c2=c2,
                                   final_target_acc=final_target_acc, target_zstd_ratio=target_zstd_ratio, 
                                   QuantizationType=QuantizationType, model=model, testloader=testloader, accuracy=accuracy, 
                                   device=device, first_best_indices=first_best_indices, accuracy_tollerance=accuracy_tollerance)
            BestQuantization_target_acc = accuracies[-1] 
        
        # ---------------------------------------------------------------------------------------------------------
        # No-pruning exit conditions
        if(pruning == "N"):
            # Entropy exit conditions
            # After the tenth epoch I must have entropy below 600000
            if(epoch >= 10 and entropies[-1] >= 600000):
                log += (
                    f"Entropy is not decreasing enough! (E1.1), r: {r}\n"
                )
                log += "-"*60

                print(log, flush = True)
                return
            
            # After the 30th epoch I must not have entropy above 200000 for 4 epochs in a row
            if(epoch >= 30):
                if(entropies[-1] >= 200000 and entropies[-2] >= 200000 and entropies[-3] >= 200000 and entropies[-4] >= 200000):
                    log += (
                        f"Entropy is not decreasing enough! (E2.1), r: {r}\n"
                    )
                    log += "-"*60

                    print(log, flush = True)
                    return
                
            # ---------------------------------------------------------------------------------------------------------
            # Accuracy exit condition
            # After the first epoch I must have accuracy above 60%
            if(epoch >= 1 and accuracies[-1] <= 60):
                log += (
                    f"Accuracy is too low! (A1.1), r: {r}\n"
                )
                log += "-"*60

                print(log, flush = True)
                return
                
            # After the 20th epoch I must have accuracy above 96%
            if(epoch >= 20 and accuracies[-1] <= 96):
                log += (
                    f"Accuracy is too low! (A1.2), r: {r}\n"
                )
                log += "-"*60

                print(log, flush = True)
                return

            # After the 80th epoch I must have accuracy above 98%
            if(epoch >= 80 and accuracies[-1] <= 98):
                log += (
                    f"Accuracy is too low! (A1.3), r: {r}\n"
                )
                log += "-"*60

                print(log, flush = True)
                return
            
            # After the 30th epoch I must not have accuracy below 90% for 4 epochs in a row
            if(epoch >= 30):
                if(accuracies[-1] <= 90 and accuracies[-2] <= 90 and accuracies[-3] <= 90 and accuracies[-4] <= 90):
                    log += (
                        f"Accuracy is too low! (A2.1), r: {r}\n"
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
                    f"Entropy is not decreasing enough! (E1.1), r: {r}\n"
                )
                log += "-"*60

                print(log, flush = True)
                return
            
            # ---------------------------------------------------------------------------------------------------------
            # Accuracy exit condition
            # After the first epoch I must have accuracy above 30%
            if(epoch >= 0 and accuracies[-1] <= 30):
                log += (
                    f"Accuracy is too low! (A1.1), r: {r}\n"
                )
                log += "-"*60

                print(log, flush = True)
                return
            """
            # After the 10th epoch I must have accuracy above 94%
            if(epoch >= 9 and accuracies[-1] <= 94):
                log += (
                    f"Accuracy is too low! (A1.2), r: {r}\n"
                )
                log += "-"*60

                print(log, flush = True, end = "")
                return
            """
            
            # ... ADD OTHER EXIT CONDITIONS IF NECESSARY...   
           
        # ---------------------------------------------------------------------------------------------------------
        
    log += "-"*60
    print(log, flush = True)
    return
