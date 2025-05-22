import argparse
import torch
import os
from utils.trainer import train_and_evaluate
from torchvision import datasets, transforms

# Function to load the MNIST dataset
def load_data():
    # Define a transformation to convert images to tensors
    transform = transforms.Compose([transforms.ToTensor()])
    
    # Load the training set of MNIST dataset with the specified transformation
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    
    # Load the test set of MNIST dataset with the specified transformation
    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # Return the loaded training and test datasets
    return trainset, testset

if __name__ == "__main__":
    # Initialize argument parser for command-line arguments
    parser = argparse.ArgumentParser()
    
    # Add argument for 'delta', which is required for the training
    parser.add_argument("--r", type=float, required=True, help="Value of r")
    
    # Parse the command-line arguments
    args = parser.parse_args()

    # Limit the number of threads used by PyTorch to 1 for CPU execution
    torch.set_num_threads(1)
    
    # Set the OpenMP number of threads to 1 for parallel processing on CPU
    os.environ["OMP_NUM_THREADS"] = "1"

    # Load the training and test datasets using the load_data function
    trainset, testset = load_data()
    
    # Create data loaders for training and testing, with specific batch sizes and no parallel data loading (num_workers=0)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False, num_workers=0)

    device = torch.device("cpu")

    # Define fixed hyperparameters for the model and training process
    C = 6
    print(f"C={C}", flush=True)
    lr = 0.0007  
    print(f"lr={lr}", flush=True)
    lambda_reg = 0.0015 
    print(f"lambda_reg={lambda_reg}", flush=True)
    alpha = 0.533  
    print(f"alpha={alpha}", flush=True)
    subgradient_step = 1e5 
    print(f"subgradient_step={subgradient_step}", flush=True)
    w0 = -0.11  
    print(f"w0={w0}", flush=True)
    #r = 1.1106  
    #print(f"r={r}", flush=True)
    target_acc = 99.00  
    print(f"target_acc={target_acc}", flush=True)
    target_entr = 0.99e5  
    print(f"target_entr={target_entr}", flush=True)
    min_xi = 0  
    print(f"min_xi={min_xi}", flush=True)
    max_xi = 1  
    print(f"max_xi={max_xi}", flush=True)
    n_epochs = 10  
    print(f"n_epochs={n_epochs}", flush=True)
    max_iterations = 15
    print(f"max_iterations={max_iterations}", flush=True)
    train_optimizer = "A"  
    print(f"train_optimizer={train_optimizer}", flush=True)
    entropy_optimizer = "F"  
    print(f"entropy_optimizer={entropy_optimizer}", flush=True)
    delta = 32
    print(f"delta={delta}", flush=True)
    pruning = "Y"
    print(f"pruning={pruning}", flush=True)
    
    train_and_evaluate(
        C=C, lr=lr, lambda_reg=lambda_reg, alpha=alpha,
        subgradient_step=subgradient_step, w0=w0, r=args.r, # Pass the value from command line arguments
        target_acc=target_acc, target_entr=target_entr,
        min_xi=min_xi, max_xi=max_xi, n_epochs=n_epochs,
        max_iterations=max_iterations,
        device=device, train_optimizer=train_optimizer,
        entropy_optimizer=entropy_optimizer,
        trainloader=trainloader, testloader=testloader,
        delta=delta, pruning=pruning 
    )
