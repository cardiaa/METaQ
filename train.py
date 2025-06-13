import argparse
import torch
import os
from utils.trainer import train_and_evaluate
from utils.networks import LeNet5
from torchvision import datasets, transforms
from torch.nn import CrossEntropyLoss

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
    model, model_name = LeNet5().to(device), "LeNet-5"
    criterion, criterion_name = CrossEntropyLoss(), "CrossEntropy" 
    C = 64
    lr = 0.0007  
    lambda_reg = 0.0015
    alpha = 0.533
    subgradient_step = 1e5 
    w0 = -0.11  
    #r = 1.1106  
    target_acc = 98.4
    target_zstd_ratio = 0.0297 
    min_xi = 0  
    max_xi = 1  
    upper_c = sum(p.numel() for p in LeNet5().parameters())
    lower_c = 1e-2
    zeta = 50000
    l = 0.5
    n_epochs = 120 # To be increased as soon as I find good configurations
    max_iterations = 15
    train_optimizer = "ADAM"  
    entropy_optimizer = "FISTA"  
    delta = 32
    pruning = "Y"
    QuantizationType = "center"

    if(args.r == 1.1001):
        print("=================================================================", flush = True)
        print("==================== PARAMETER CONFIGURATION ====================", flush = True)
        print("=================================================================", flush = True)
        print(f"model={model_name}", flush=True)
        print(f"criterion={criterion_name}", flush=True)
        print(f"C={C}", flush=True)
        print(f"lr={lr}", flush=True)    
        print(f"lambda_reg={lambda_reg}", flush=True)
        print(f"alpha={alpha}", flush=True)    
        print(f"[T1=lambda_reg*alpha={round(lambda_reg*alpha, 6)}]", flush=True)
        print(f"[T2=lambda_reg*(1-alpha)={round(lambda_reg*(1-alpha), 6)}]", flush=True)
        print(f"subgradient_step={subgradient_step}", flush=True)    
        print(f"w0={w0}", flush=True)    
        #print(f"r={r}", flush=True)    
        print(f"target_acc={target_acc}", flush=True)    
        print(f"target_zstd_ratio={target_zstd_ratio}", flush=True)    
        print(f"min_xi={min_xi}", flush=True)    
        print(f"max_xi={max_xi}", flush=True)  
        print(f"upper_c={upper_c}", flush=True)
        print(f"lower_c={lower_c}", flush=True)  
        print(f"zeta={zeta}", flush=True)
        print(f"l={l}", flush=True)
        print(f"n_epochs={n_epochs}", flush=True) 
        print(f"max_iterations={max_iterations}", flush=True)    
        print(f"train_optimizer={train_optimizer}", flush=True)    
        print(f"entropy_optimizer={entropy_optimizer}", flush=True) 
        print(f"delta={delta}", flush=True) 
        print(f"pruning={pruning}", flush=True)
        print(f"QuantizationType={QuantizationType}", flush=True)
        print("-"*60, flush=True)
    
    train_and_evaluate(
        model=model, criterion=criterion, C=C, lr=lr, lambda_reg=lambda_reg, alpha=alpha,
        subgradient_step=subgradient_step, w0=w0, r=args.r, # Pass the value from command line arguments
        target_acc=target_acc, target_zstd_ratio=target_zstd_ratio,
        min_xi=min_xi, max_xi=max_xi, upper_c=upper_c, lower_c=lower_c, zeta=zeta, l=l, n_epochs=n_epochs,
        max_iterations=max_iterations,
        device=device, train_optimizer=train_optimizer,
        entropy_optimizer=entropy_optimizer,
        trainloader=trainloader, testloader=testloader,
        delta=delta, pruning=pruning, QuantizationType=QuantizationType
    )
