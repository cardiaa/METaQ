import torch  
import time
import numpy as np
from torchvision import datasets, transforms  
from itertools import product
from utils.trainer import train_and_evaluate  

# Main entry point of the script
if __name__ == "__main__":

    # Select the computing device: use GPU if available, otherwise fallback to CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Define a transformation: convert images to tensors
    transform = transforms.Compose([transforms.ToTensor()])
    # Load the MNIST training dataset with the defined transformation
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    # Create a DataLoader for the training set with batch size 64, shuffling enabled, and 4 worker threads
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)
    # Load the MNIST test dataset with the same transformation
    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    # Create a DataLoader for the test set with batch size 1000, shuffling disabled, and 4 worker threads
    testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False, num_workers=4)
    
    np.set_printoptions(precision=6)

    # Grid search 
    param_grid = {
        "C": [6, 12, 24, 48],  # Number of buckets of quantization
        "lr": [0.0007], # Learning rate for the optimizer
        "lambda_reg": [0.0015], # Regularization factor
        "alpha": [0.533], # Percentage of standard regularization wrt entropic one 
        "subgradient_step": [1e5],  # Step size for subgradient
        "w0": [-0.11], # Initial weight parameters
        "r": [round(1.1 + i * 0.002, 3) for i in range(10)],
        "target_acc": [98.99], # Target accuracy percentage
        "target_entr": [0.99602e6], # Target entropy threshold 
        "min_xi": [0], # lower bound for xi initialization
        "max_xi": [1],  # upper bound for xi initialization
        "n_epochs": [100], # Number of training epochs
        "device": [device], # Computing device (GPU or CPU)
        "train_optimizer": ['A'],  # 'A' for Adam, and 'S' for SGD
        "entropy_optimizer": ['F'], # 'F' for FISTA, 'PM' for proximal bundle
        "trainloader": [trainloader],  # Training data loader
        "testloader": [testloader] # Test data loader
    }

    combination = 0

    for (C, lr, lambda_reg, alpha, subgradient_step, w0, r, 
        target_acc, target_entr, min_xi, max_xi, n_epochs, 
        device, train_optimizer, entropy_optimizer, trainloader, 
        testloader) in product(param_grid["C"],
                                param_grid["lr"],
                                param_grid["lambda_reg"],
                                param_grid["alpha"],
                                param_grid["subgradient_step"],
                                param_grid["w0"],
                                param_grid["r"],
                                param_grid["target_acc"],
                                param_grid["target_entr"],
                                param_grid["min_xi"],
                                param_grid["max_xi"],
                                param_grid["n_epochs"],
                                param_grid["device"],
                                param_grid["train_optimizer"],      
                                param_grid["entropy_optimizer"],   
                                param_grid["trainloader"], 
                                param_grid["testloader"]
                                ):
        
        # Counts combinations
        combination += 1
        
        # Start training
        start_time = time.time()
        accuracy, entropy, target_acc, target_entr = train_and_evaluate(C=C,              
                                                                    lr=lr,           
                                                                    lambda_reg=lambda_reg,    
                                                                    alpha=alpha,          
                                                                    subgradient_step=subgradient_step, 
                                                                    w0=w0,             
                                                                    r=r,              
                                                                    target_acc=target_acc,      
                                                                    target_entr=target_entr, 
                                                                    min_xi=min_xi,              
                                                                    max_xi=max_xi,             
                                                                    n_epochs=n_epochs,        
                                                                    device=device,      
                                                                    train_optimizer=train_optimizer,     
                                                                    entropy_optimizer=entropy_optimizer,   
                                                                    trainloader=trainloader, 
                                                                    testloader=testloader     
                                                                )
            
        training_time = time.time() - start_time
        print(f'Time spent to train the model: {training_time:.2f} seconds\n')





















