import torch
import time 
import numpy as np
import os
import torch.nn as nn
import torch.optim as optim
import copy
import struct
import math
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from collections import Counter
from utils.networks import LeNet5
from utils.quantize_and_compress import compute_entropy, compute_entropy_new, quantize_weights_center
from utils.optimization import FISTA, ProximalBM
from utils.weight_utils import initialize_weights
from utils.quantize_and_compress import compare_lists, compress_zstd, decompress_zstd

def test_accuracy(model, dataloader, device):
    """
    Function to calculate the accuracy of a model on a given dataloader.
    """
    correct, total = 0, 0
    with torch.no_grad():  # Disable gradient computation for evaluation
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)  # Move data to the appropriate device
            outputs = model(images)  # Get model predictions
            _, predicted = torch.max(outputs.data, 1)  # Get the class with the highest probability
            total += labels.size(0)  # Update total number of samples
            correct += (predicted == labels).sum().item()  # Count correct predictions
    
    accuracy = 100 * correct / total  # Compute accuracy percentage
    return accuracy


def train_and_evaluate(C, lr, lambda_reg, alpha, subgradient_step, w0, r,
                        target_acc, target_entr, min_xi, max_xi, n_epochs,
                        max_iterations, device, train_optimizer, entropy_optimizer, 
                        trainloader, testloader, delta, pruning):
    
    torch.set_num_threads(1)

    # Initialization of the model, loss function, and optimizer.
    model = LeNet5().to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Selection of the optimizer based on the chosen type.
    if train_optimizer == 'A':
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=lambda_reg * alpha)
    elif train_optimizer == 'S':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=lambda_reg * alpha)
    
    # Weights Initialization
    min_w, max_w = w0 - r, w0 + r
    v = torch.linspace(min_w, max_w - (max_w - min_w)/C, steps=C, device=device)
    initialize_weights(model, min_w, max_w)
    w = torch.cat([param.data.view(-1) for param in model.parameters()]).to(device)
    upper_c, lower_c = w.size(0), 1e-2
    xi = min_xi + (max_xi - min_xi) * torch.rand(C, device=device)
    xi = torch.sort(xi)[0]   
    entropy, accuracy = 0, 0
    accuracies, entropies, distinct_weights = [], [], []
    zeta, l = 50000, 0.5

    print("... Training Started ...", flush=True)
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
            if(entropy_optimizer == 'F'):
                #xi, beta_tensor, x_star, phi = FISTA(xi, v, w_quantized, C, subgradient_step, max_iterations, pruning) # Alternative version
                xi, beta_tensor, x_star, phi = FISTA(xi, v, w, C, delta, subgradient_step, device, max_iterations, pruning) 
            elif(entropy_optimizer == 'PM'):
                #xi, beta_tensor, x_star, phi = ProximalBM(xi, v, w_quantized, C, zeta, subgradient_step, max_iterations, pruning) # Alternative version
                xi, beta_tensor, x_star, phi = ProximalBM(xi, v, w, C, delta, zeta, subgradient_step, device, max_iterations, pruning)       

            # Update of ∇ɸ
            idx = 0
            for param in model.parameters():
                numel = param.numel()
                if param.grad is not None:
                    param_grad = param.grad.view(-1)
                else:
                    param_grad = torch.zeros_like(param.data.view(-1)).to(device)
                param_grad += (1 - alpha) * lambda_reg * beta_tensor[idx:idx + numel]
                param.grad = param_grad.view(param.size())
                idx += numel
            
            loss.backward()
            optimizer.step()
        
        w = torch.cat([param.data.view(-1) for param in model.parameters()]).to(device)
        
        target_entr_before = 1e4
        pruning_threshold = 1e-4
        
        accuracy = test_accuracy(model, testloader, device)
        accuracies.append(accuracy)
        
        entropy = round(compute_entropy(w.tolist())) + 1
        entropies.append(entropy)
        #print(f"entropy={entropy}")

        v_centers_before = (v[:-1] + v[1:]) / 2
        v_centers_before = torch.cat([v_centers_before, v[-1:]])
        w_quantized_before = quantize_weights_center(w, v, v_centers_before)
        encoded_list_before = [float(elem) if float(elem) != -0.0 else 0.0 for elem in w_quantized_before]
        quantized_entropy_before = round(compute_entropy(encoded_list_before)) + 1
        #print(f"quantized_entropy_before={quantized_entropy_before}")

        entropy_new_formula = round(compute_entropy_new(w.tolist(), pruning_threshold)) + 1
        #print(f"entropy_new_formula={entropy_new_formula}")

        quantized_entropy_new_formula = round(compute_entropy_new(encoded_list_before, pruning_threshold)) + 1
        #print(f"quantized_entropy_new_formula={quantized_entropy_new_formula}")

        training_time = time.time() - start_time
        print(f"Epoca {epoch + 1}: lr = {lr}, Accuracy = {accuracies[-1]}, H_NQ = {entropies[-1]}, H_Q = {quantized_entropy_before}, "
              f"H_NQ_new = {entropy_new_formula}, H_Q_new = {quantized_entropy_new_formula}", flush = True)
        
        # Saving a better model
        #if(entropies[-1] <= target_entr):
        if(accuracies[-1] >= 93):
            c1=10
            c2=1000
            QuantAcc = []
            QuantEntr = []
            #print("... Looking for the best quantization ...")
            # Test quantization in C in [10, 1000] buckets
            for C_tmp in range(c1, c2 + 1):
                # Compute central values of the buckets
                v_tmp = torch.linspace(min_w, max_w - (max_w - min_w)/C_tmp, steps=C_tmp)
                v_centers = (v_tmp[:-1] + v_tmp[1:]) / 2
                v_centers = torch.cat([v_centers, v_tmp[-1:]])  # Add final value to handle the last bucket
                # Quantize weights using central values
                w_quantized = quantize_weights_center(w, v_tmp, v_centers)
                model_quantized = copy.deepcopy(model).to(device)
                # Replace quantized weights in the quantized model
                start_idx = 0
                for param in model_quantized.parameters():
                    numel = param.data.numel()
                    param.data = w_quantized[start_idx:start_idx + numel].view(param.data.size())
                    start_idx += numel
                # Evaluate quantized model
                model_quantized.eval()
                num_unique_weights_quantized = torch.unique(w_quantized).numel()
                quantized_accuracy = test_accuracy(model_quantized, testloader, device)
                # Compute entropy of the quantized string
                encoded_list = [float(elem) if float(elem) != -0.0 else 0.0 for elem in w_quantized]
                quantized_entropy = round(compute_entropy(encoded_list)) + 1
                QuantAcc.append(quantized_accuracy)
                QuantEntr.append(quantized_entropy)
            # Print results for the best 10 models
            sorted_indices = np.argsort(QuantAcc)
            for i in range(1, 10):
                C_tmp = sorted_indices[-i] + c1
                v_tmp = torch.linspace(min_w, max_w - (max_w - min_w)/C_tmp, steps=C_tmp)
                v_centers = (v_tmp[:-1] + v_tmp[1:]) / 2
                v_centers = torch.cat([v_centers, v_tmp[-1:]])
                model_quantized = copy.deepcopy(model).to(device)
                # Extract model weights
                w_saved = torch.cat([param.data.view(-1) for param in model_quantized.parameters()])
                # Quantize weights using central values
                w_quantized = quantize_weights_center(w_saved, v_tmp, v_centers)
                encoded_list = [float(elem) if float(elem) != -0.0 else 0.0 for elem in w_quantized]

                quantized_entropy = round(compute_entropy(encoded_list)) + 1
                quantized_entropy_new_formula = round(compute_entropy_new(encoded_list, pruning_threshold)) + 1

                # Converts float list in byte
                input_bytes = b''.join(struct.pack('f', num) for num in encoded_list)
                # Compression
                zstd_compressed = compress_zstd(input_bytes)
                # Decompression
                zstd_decompressed = decompress_zstd(zstd_compressed)
                # Verifies
                if not compare_lists(encoded_list, zstd_decompressed):
                    print(f"💥💥💥 Encoding error! Decoded 💥💥💥")                    
                # Calculates dimensions
                original_size_bits = len(input_bytes) * 8
                zstd_size = len(zstd_compressed) * 8
                # Compression ratio
                zstd_ratio = zstd_size / original_size_bits
                # Output delle dimensioni e del rapporto di compressione
                if(QuantAcc[sorted_indices[-i]] >= target_acc and zstd_ratio <= 0.0343):
                    torch.save(model.state_dict(), f"BestModelsMay2025/Test2May2025_C{C}_r{r}_epoch{epoch}.pth")
                    print("✅"*50, flush = True)
                    print("✅"*50, flush = True)
                    print("✅"*50, flush = True)
                    print("✅✅✅✅✅✅ MODEL SAVED ✅✅✅✅✅✅", flush=True)
                    print("✅"*50, flush = True)
                    print("✅"*50, flush = True)
                    print("✅"*50, flush = True)
                #if(zstd_ratio <= 0.0343):
                if(True):
                    #print("💥"*50)
                    #print("💥💥💥 ...AIN'T SAVING THE MODEL... JUST CHECKING... 💥💥💥")
                    #print("💥"*50)
                    #print("💥"*50)
                    #print(f"💥💥💥 r={r}, pruning={pruning}, Quantization at C={sorted_indices[-i] + c1}, "
                    print(f"\t➡️ lr = {lr}, Quantization at C={sorted_indices[-i] + c1}, Accuracy from {accuracy} to {QuantAcc[sorted_indices[-i]]}, H_Q_new = {quantized_entropy_new_formula}", flush = True)
                    #print(f"💥💥💥 Entropy: from {entropy} to {QuantEntr[sorted_indices[-i]]} (standard formula) 💥💥💥")
                    #print(f"💥💥💥 quantized_entropy_new_formula={quantized_entropy_new_formula} (new formula) 💥💥💥")
                    #print(f"💥💥💥 Original dimension: {original_size_bits} bits 💥💥💥")
                    print(f"\tZstd-22 compressed dimension = {zstd_size} bits (Compression Ratio = {zstd_ratio:.2%})", flush = True)
                    #print("💥"*50)
                    #print("💥"*50)
                    #print("💥"*50)
        
        # No-pruning exit conditions
        if(pruning == "N"):
            # Entropy exit conditions
            # After the tenth epoch I must have entropy below 600000
            if(epoch >= 10 and entropies[-1] >= 600000):
                print(f"Entropy is not decreasing enough! (E1.1), PID: {os.getpid()}, Epoch: {epoch}, "
                    f"Current Entropy: {entropies[-1]}, Current Accuracy: {accuracies[-1]}, "
                    f"C: {C}, r: {r}, epoch time: {training_time:.2f}s", flush=True)
                return accuracies[-1], entropies[-1], target_acc, target_entr
            
            # After the 30th epoch I must not have entropy above 200000 for 4 epochs in a row
            if(epoch >= 30):
                if(entropies[-1] >= 200000 and entropies[-2] >= 200000 and entropies[-3] >= 200000 and entropies[-4] >= 200000):
                    print(f"Entropy is not decreasing enough! (E2.1), PID: {os.getpid()}, Epoch: {epoch}, "
                        f"Current Entropy: {entropies[-1]}, Current Accuracy: {accuracies[-1]}, "
                        f"C: {C}, r: {r}, epoch time: {training_time:.2f}s", flush=True)
                    return accuracies[-1], entropies[-1], target_acc, target_entr           
                
            # ---------------------------------------------------------------------------------------------------------
            # Accuracy exit condition
            # After the first epoch I must have accuracy above 60%
            if(epoch >= 1 and accuracies[-1] <= 60):
                print(f"Accuracy is too low! (A1.1), PID: {os.getpid()}, Epoch: {epoch}, "
                    f"Current Entropy: {entropies[-1]}, Current Accuracy: {accuracies[-1]}, "
                    f"C: {C}, r: {r}, epoch time: {training_time:.2f}s", flush=True)
                return accuracies[-1], entropies[-1], target_acc, target_entr  
                
            # After the 20th epoch I must have accuracy above 96%
            if(epoch >= 20 and accuracies[-1] <= 96):
                print(f"Accuracy is too low! (A1.2), PID: {os.getpid()}, Epoch: {epoch}, "
                    f"Current Entropy: {entropies[-1]}, Current Accuracy: {accuracies[-1]}, "
                    f"C: {C}, r: {r}, epoch time: {training_time:.2f}s", flush=True)
                return accuracies[-1], entropies[-1], target_acc, target_entr

            # After the 80th epoch I must have accuracy above 98%
            if(epoch >= 80 and accuracies[-1] <= 98):
                print(f"Accuracy is too low! (A1.3), PID: {os.getpid()}, Epoch: {epoch}, "
                    f"Current Entropy: {entropies[-1]}, Current Accuracy: {accuracies[-1]}, "
                    f"C: {C}, r: {r}, epoch time: {training_time:.2f}s", flush=True)
                return accuracies[-1], entropies[-1], target_acc, target_entr
            
            # After the 30th epoch I must not have accuracy below 90% for 4 epochs in a row
            if(epoch >= 30):
                if(accuracies[-1] <= 90 and accuracies[-2] <= 90 and accuracies[-3] <= 90 and accuracies[-4] <= 90):
                    print(f"Accuracy is too low! (A2.1), PID: {os.getpid()}, Epoch: {epoch}, "
                        f"Current Entropy: {entropies[-1]}, Current Accuracy: {accuracies[-1]}, "
                        f"C: {C}, r: {r}, epoch time: {training_time:.2f}s", flush=True)
                    return accuracies[-1], entropies[-1], target_acc, target_entr     
            
            # ... ADD OTHER EXIT CONDITIONS IF NECESSARY...   
        # ---------------------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------------------------------
        # Pruning exit conditions
        elif(pruning == "-"):
            # Accuracy exit condition
            # After the first epoch I must have accuracy above 60%
            if(epoch >= 1 and accuracies[-1] <= 60):
                print(f"Accuracy is too low! (A1.1), PID: {os.getpid()}, Epoch: {epoch}, "
                    f"Current Entropy: {entropies[-1]}, Current Accuracy: {accuracies[-1]}, "
                    f"C: {C}, r: {r}, epoch time: {training_time:.2f}s", flush=True)
                return accuracies[-1], entropies[-1], target_acc, target_entr  
                
            # After the 20th epoch I must have accuracy above 95%
            if(epoch >= 20 and accuracies[-1] <= 95):
                print(f"Accuracy is too low! (A1.2), PID: {os.getpid()}, Epoch: {epoch}, "
                    f"Current Entropy: {entropies[-1]}, Current Accuracy: {accuracies[-1]}, "
                    f"C: {C}, r: {r}, epoch time: {training_time:.2f}s", flush=True)
                return accuracies[-1], entropies[-1], target_acc, target_entr

            # After the 50th epoch I must have accuracy above 97%
            #if(epoch >= 50 and accuracies[-1] <= 97):
            #    print(f"Accuracy is too low! (A1.3), PID: {os.getpid()}, Epoch: {epoch}, "
            #        f"Current Entropy: {entropies[-1]}, Current Accuracy: {accuracies[-1]}, "
            #        f"C: {C}, r: {r}, epoch time: {training_time:.2f}s", flush=True)
            #    return accuracies[-1], entropies[-1], target_acc, target_entr
            
            # After the 80th epoch I must have accuracy above 98%
            #if(epoch >= 80 and accuracies[-1] <= 98):
            #    print(f"Accuracy is too low! (A1.4), PID: {os.getpid()}, Epoch: {epoch}, "
            #        f"Current Entropy: {entropies[-1]}, Current Accuracy: {accuracies[-1]}, "
            #        f"C: {C}, r: {r}, epoch time: {training_time:.2f}s", flush=True)
            #    return accuracies[-1], entropies[-1], target_acc, target_entr
            
            # After the 10th epoch I must not have accuracy below 90% for 4 epochs in a row
            if(epoch >= 10):
                if(accuracies[-1] <= 90 and accuracies[-2] <= 90 and accuracies[-3] <= 90 and accuracies[-4] <= 90):
                    print(f"Accuracy is too low! (A2.1), PID: {os.getpid()}, Epoch: {epoch}, "
                        f"Current Entropy: {entropies[-1]}, Current Accuracy: {accuracies[-1]}, "
                        f"C: {C}, r: {r}, epoch time: {training_time:.2f}s", flush=True)
                    return accuracies[-1], entropies[-1], target_acc, target_entr     
            
            # ... ADD OTHER EXIT CONDITIONS IF NECESSARY...   
           
        # ---------------------------------------------------------------------------------------------------------
        
        """
        print(f"➡️ r: {r}, C: {C}, Epoch: {epoch}, Current Entropy: {entropies[-1]}, Current Accuracy: {accuracies[-1]}, "
              f"quantized_entropy_before: {quantized_entropy_before}, entropy_new_formula: {entropy_new_formula}, "
              f"Min Entropy: {min(entropies)}, Max Accuracy: {max(accuracies)}, pruning: {pruning}, delta: {delta}, "
              f"epoch time: {training_time:.2f}s, N_zeroes: {(w == 0).sum().item()}, " 
              f"Percent_zeroes: {(w == 0).float().mean().item() * 100}, N_under_threshold: {(w <= pruning_threshold).sum().item()}, "
              f"Percent_under_threshold: {(w <= pruning_threshold).float().mean().item() * 100}\n", flush=True)
        """
        print("-"*60, flush = True)

    return accuracies[-1], entropies[-1], target_acc, target_entr
