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

    # Define fixed hyperparameters for the model and training process
    C = 6  
    lr = 0.0007  
    lambda_reg = 0.0015 
    alpha = 0.533  
    subgradient_step = 1e5 
    w0 = -0.11  
    #r = 1.1  
    target_acc = 99.00  
    target_entr = 1.2e5  
    min_xi = 0  
    max_xi = 1  
    n_epochs = 1000  
    max_iterations = 15
    device = torch.device("cpu")  
    train_optimizer = "A"  
    entropy_optimizer = "F"  
    delta = 0
    pruning = "Y"
    
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
