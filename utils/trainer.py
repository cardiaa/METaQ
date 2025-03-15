import torch
import time 
import numpy as np
import os
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from utils.networks import LeNet5
from utils.quantize_and_compress import compute_entropy
from utils.optimization import FISTA, ProximalBM
from utils.weight_utils import initialize_weights
from IPython.display import clear_output

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
                       target_acc, target_entr, min_xi, max_xi, n_epochs, device, 
                       train_optimizer, entropy_optimizer, trainloader, testloader):
    
    model = LeNet5().to(device)
    criterion = nn.CrossEntropyLoss()
    
    if(train_optimizer == 'A'):
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=lambda_reg * alpha)
    elif(train_optimizer == 'S'):
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=lambda_reg * alpha)
    
    # Parameters initialization
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
    
    for epoch in range(n_epochs):
        start_time = time.time()
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            w = torch.cat([param.data.view(-1) for param in model.parameters()])
            #unique_weights = torch.unique(w).numel() 
            #indices = torch.searchsorted(v, w, right=True) - 1
            #indices = torch.clamp(indices, min=0)
            #w_quantized = v[indices]

            zeta *= 1 + l
            l = l / 1.5
            if(entropy_optimizer == 'F'):
                #xi, beta_tensor, x_star, phi = FISTA(xi, v, w_quantized, C, subgradient_step, max_iterations=15) 
                xi, beta_tensor, x_star, phi = FISTA(xi, v, w, C, subgradient_step, device, max_iterations=15) 
            elif(entropy_optimizer == 'PM'):
                #xi, beta_tensor, x_star, phi = ProximalBM(xi, v, w_quantized, C, zeta, subgradient_step, max_iterations=15) 
                xi, beta_tensor, x_star, phi = ProximalBM(xi, v, w, C, zeta, subgradient_step, device, max_iterations=15)       
            
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
        
        entropy = round(compute_entropy(w.tolist())) + 1
        entropies.append(entropy)
        accuracy = test_accuracy(model, testloader, device)
        accuracies.append(accuracy)
        
        # Creo un file di log per ogni combinazione
        #output_dir = "training_logs"
        #os.makedirs(output_dir, exist_ok=True)
        #log_filename = f"{output_dir}/log_C_{C}_r_{r}_proc_{os.getpid()}.txt"

        #with open(log_filename, "a") as f:
        #    f.write("\nEpoch:", epoch+1)
        #    f.write("\nAccuracies:", accuracies)
        #    f.write("\nEntropies:", entropies)
        #    f.write("\nMax Accuracy:", max(accuracies))
        #    f.write("Min entropy:", min(entropies))

        # Saving a better model
        if(accuracy >= target_acc and entropy <= target_entr):
            #with open(log_filename, "a") as f:
            #    f.write("💥💥💥💥💥💥💥\n💥ATTENTION!💥\n💥💥💥💥💥💥💥")
            print("💥💥💥💥💥💥💥\n💥ATTENTION!💥\n💥💥💥💥💥💥💥")
            torch.save(model.state_dict(), f"BestModelsBeforeQuantization/C{C}_r{round(r*1000)}.pth")
            target_acc = accuracy
            target_entr = entropy
        
        # Entropy exit conditions
        if(epoch > 20 and entropy > 600000):
            #with open(log_filename, "a") as f:
            #    f.write("Entropy is not decreasing enough! (A)")
            print("Entropy is not decreasing enough! (A), PID: {os.getpid()}, Epoch: {epoch}, Entropia minima: {min(entropies)}, Accuracy massima: {max(accuracies)}, C: {C}, r: {r}, epoch time: {training_time:.2f}s")
            return accuracy, entropy, target_acc, target_entr
        
        if(epoch > 50):
            if(entropies[-1] > 200000 and entropies[-2] > 200000 and entropies[-3] > 200000 and entropies[-4] > 200000):
                #with open(log_filename, "a") as f:
                #    f.write("Entropy is not decreasing enough! (B)")
                print("Entropy is not decreasing enough! (B), PID: {os.getpid()}, Epoch: {epoch}, Entropia minima: {min(entropies)}, Accuracy massima: {max(accuracies)}, C: {C}, r: {r}, epoch time: {training_time:.2f}s")
                return accuracy, entropy, target_acc, target_entr           
            
        # Accuracy exit condition
        if(epoch == 1 and accuracies[-1] < 70):
            #with open(log_filename, "a") as f:
            #    f.write("Accuracy is too low! (C)")
            print("Accuracy is too low! (C), PID: {os.getpid()}, Epoch: {epoch}, Entropia minima: {min(entropies)}, Accuracy massima: {max(accuracies)}, C: {C}, r: {r}, epoch time: {training_time:.2f}s")
            return accuracy, entropy, target_acc, target_entr  
                          
        if(epoch > 10):
            if(accuracies[-1] < 90 and accuracies[-2] < 90 and accuracies[-3] < 90 and accuracies[-4] < 90):
                #with open(log_filename, "a") as f:
                #    f.write("Accuracy is too low! (D)")
                print("Accuracy is too low! (D), PID: {os.getpid()}, Epoch: {epoch}, Entropia minima: {min(entropies)}, Accuracy massima: {max(accuracies)}, C: {C}, r: {r}, epoch time: {training_time:.2f}s")
                return accuracy, entropy, target_acc, target_entr     
        
        # ... ADD OTHER EXIT CONDITIONS ...      
        
        training_time = time.time() - start_time
        #with open(log_filename, "a") as f:
        #    f.write(f"Time taken for a epoch: {training_time:.2f} seconds\n")
              
        print(f"PID: {os.getpid()}, Epoch: {epoch}, Entropia minima: {min(entropies)}, Accuracy massima: {max(accuracies)}, C: {C}, r: {r}, epoch time: {training_time:.2f}s")

    return accuracy, entropy, target_acc, target_entr
