import argparse
import torch
import os
import torch.distributed as dist
from utils.trainer_on_gpus import train_and_evaluate
from utils.networks import LeNet5, LeNet5_enhanced, LeNet5_Original, LeNet300_100
from torchvision import datasets, transforms, models
import torch.nn as nn 
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

def setup():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_world_size()

def cleanup():
    dist.destroy_process_group()

# Function to load the MNIST dataset
def load_data(model_name):

    if(model_name[:7] == "LeNet-5"):
        if(model_name[-9:] == "(rotated)"):
            # Data Augmentation + Resizing images to 32x32
            transform_train = transforms.Compose([
                transforms.Resize(32),
                transforms.RandomRotation(10),
                transforms.RandomAffine(0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
            ])

            transform_test = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
            ])
            # Load the training set of MNIST dataset with the specified transformation
            trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
            # Load the test set of MNIST dataset with the specified transformation
            testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)     
        else:
            # No Data Augmentation + Resizing images to 32x32
            transform = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
            ])
            # Load the training set of MNIST dataset with the specified transformation
            trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
            # Load the test set of MNIST dataset with the specified transformation
            testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)   
    elif(model_name[:12] == "LeNet300_100"):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # Media e dev. std di MNIST
        ])
        trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    elif(model_name == "AlexNet"):
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                [0.229, 0.224, 0.225])
        ])

        transform_val = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                [0.229, 0.224, 0.225])
        ])   

        train_dataset = datasets.ImageFolder('/disk1/a.cardia/imagenet/train', transform=transform_train)
        val_dataset = datasets.ImageFolder('/disk1/a.cardia/imagenet/val', transform=transform_val)

        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank, shuffle=True, drop_last=True)

        trainset = DataLoader(train_dataset, batch_size=1024, sampler=train_sampler, num_workers=8, pin_memory=True)
        testset = DataLoader(val_dataset, batch_size=1024, shuffle=False, num_workers=8, pin_memory=True)
    elif(model_name == "VGG16"):
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                [0.229, 0.224, 0.225])
        ])

        transform_val = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                [0.229, 0.224, 0.225])
        ])   

        train_dataset = datasets.ImageFolder('/disk1/a.cardia/imagenet/train', transform=transform_train)
        val_dataset = datasets.ImageFolder('/disk1/a.cardia/imagenet/val', transform=transform_val)

        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank, shuffle=True, drop_last=True)

        trainset = DataLoader(train_dataset, batch_size=512, sampler=train_sampler, num_workers=8, pin_memory=True) #with 512 and C=4 it works
        testset = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=8, pin_memory=True)        

    # Return the loaded training and test datasets
    if(model_name == "AlexNet" or model_name == "VGG16"):
        return trainset, testset, train_sampler
    else:
        return trainset, testset

if __name__ == "__main__":
    # Initialize argument parser for command-line arguments
    parser = argparse.ArgumentParser()
    # Add argument for 'delta', which is required for the training
    parser.add_argument("--delta", type=float, required=True, help="Value of delta")
    # Add argument for model name, so the user can choose the architecture from the command line
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=["LeNet-5", "LeNet-5 (rotated)", "LeNet300_100", "AlexNet", "VGG16"],
        help="Name of the model to train"
    )
    # Parse the command-line arguments
    args = parser.parse_args()

    # Limit the number of threads used by PyTorch to 1 for CPU execution
    torch.set_num_threads(1)
    # Set the OpenMP number of threads to 1 for parallel processing on CPU
    os.environ["OMP_NUM_THREADS"] = "1"

    # Use model name from command line instead of hardcoding
    model_name = args.model_name

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        if(model_name != "AlexNet" or model_name != "VGG16"):
            print(f"Using device {device} ({torch.cuda.get_device_name(device)})", flush=True)
    else:
        device = torch.device("cpu")

    if(model_name[:7] == "LeNet-5"):
        model = LeNet5_Original()
        model = model.to(device)         
        criterion, criterion_name = nn.CrossEntropyLoss(), "CrossEntropy" 
        C = 64
        lr = 0.001
        lambda_reg = 0.0002
        alpha = 0.6
        subgradient_step = 1e5 
        bucket_zero = round((C-1)/2) #it must range from 0 to C-2
        r = 2
        w0 = round(r - (bucket_zero + 0.5) * 2 * r * (1 - 1/C) / (C - 1), 3)
        BestQuantization_target_acc = 99.8
        final_target_acc = 99.7
        target_zstd_ratio = 0.0179
        min_xi = 0  
        max_xi = 1  
        upper_c = sum(p.numel() for p in LeNet5().parameters())
        lower_c = 1e-2
        c1 = 10
        c2 = 1000
        first_best_indices = 20
        accuracy_tollerance = 0.2
        zeta = 50000
        l = 0.5
        n_epochs = 100 # To be increased as soon as I find good configurations
        max_iterations = 15
        train_optimizer = "ADAM"  
        entropy_optimizer = "FISTA"  
        pruning = "Y"
        QuantizationType = "center"
        sparsity_threshold = 1e-3
    elif(model_name == "LeNet300_100"):
        model = LeNet300_100()
        model = model.to(device)            
        criterion, criterion_name = nn.CrossEntropyLoss(), "CrossEntropy" 
        C = 64
        lr = 0.001
        lambda_reg = 0.0002
        alpha = 0.6
        subgradient_step = 1e5 
        bucket_zero = round((C-1)/2) #it must range from 0 to C-2
        r = 2
        w0 = round(r - (bucket_zero + 0.5) * 2 * r * (1 - 1/C) / (C - 1), 3)
        BestQuantization_target_acc = 99.8
        final_target_acc = 99.7
        target_zstd_ratio = 0.0179
        min_xi = 0  
        max_xi = 1  
        upper_c = sum(p.numel() for p in LeNet300_100().parameters())
        lower_c = 1e-2
        c1 = 10
        c2 = 1000
        first_best_indices = 20
        accuracy_tollerance = 0.2
        zeta = 50000
        l = 0.5
        n_epochs = 100 # To be increased as soon as I find good configurations
        max_iterations = 15
        train_optimizer = "ADAM"  
        entropy_optimizer = "FISTA"  
        pruning = "Y"
        QuantizationType = "center"
        sparsity_threshold = 1e-3  
    elif(model_name == "AlexNet"):
        local_rank, world_size = setup()
        device = torch.device(f"cuda:{local_rank}")
        print(f"[GPU {local_rank}] Using device {device} ({torch.cuda.get_device_name(device)})", flush=True)        
        model = models.alexnet(weights=None)
        model.classifier[6] = nn.Linear(4096, 1000)
        model = model.to(device)  
        model = DDP(model, device_ids=[local_rank])     
        criterion, criterion_name = nn.CrossEntropyLoss(), "CrossEntropy" 
        C = 64
        lr = 0.01
        lambda_reg = 0.0005
        alpha = 1
        subgradient_step = 1e5 
        bucket_zero = round((C-1)/2) #it must range from 0 to C-2
        r = 2
        w0 = round(r - (bucket_zero + 0.5) * 2 * r * (1 - 1/C) / (C - 1), 3)
        BestQuantization_target_acc = 99.8
        final_target_acc = 99.7
        target_zstd_ratio = 0.0179
        min_xi = 0  
        max_xi = 1  
        upper_c = sum(p.numel() for p in model.parameters())
        lower_c = 1e-2
        c1 = 10
        c2 = 1000
        first_best_indices = 20
        accuracy_tollerance = 0.2
        zeta = 50000
        l = 0.5
        n_epochs = 20 # To be increased as soon as I find good configurations
        max_iterations = 15
        train_optimizer = "SGD"  
        entropy_optimizer = "FISTA"  
        pruning = "Y"
        QuantizationType = "center"
        sparsity_threshold = 1e-3  
    elif(model_name == "VGG16"):
        local_rank, world_size = setup()
        device = torch.device(f"cuda:{local_rank}")
        print(f"[GPU {local_rank}] Using device {device} ({torch.cuda.get_device_name(device)})", flush=True)        
        model = models.vgg16(weights=None)
        model.classifier[6] = nn.Linear(4096, 1000)
        model = model.to(device)  
        model = DDP(model, device_ids=[local_rank])     
        criterion, criterion_name = nn.CrossEntropyLoss(), "CrossEntropy" 
        C = 8
        lr = 0.01
        lambda_reg = 0.0005
        alpha = 1
        subgradient_step = 1e5 
        bucket_zero = round((C-1)/2) #it must range from 0 to C-2
        r = 2
        w0 = round(r - (bucket_zero + 0.5) * 2 * r * (1 - 1/C) / (C - 1), 3)
        BestQuantization_target_acc = 99.8
        final_target_acc = 99.7
        target_zstd_ratio = 0.0179
        min_xi = 0  
        max_xi = 1  
        upper_c = sum(p.numel() for p in model.parameters())
        lower_c = 1e-2
        c1 = 10
        c2 = 1000
        first_best_indices = 20
        accuracy_tollerance = 0.2
        zeta = 50000
        l = 0.5
        n_epochs = 20 # To be increased as soon as I find good configurations
        max_iterations = 15
        train_optimizer = "SGD"  
        entropy_optimizer = "FISTA"  
        pruning = "Y"
        QuantizationType = "center"
        sparsity_threshold = 1e-3          

    # Only print parameters from the first process/GPU
    if(model_name == "AlexNet" or model_name == "VGG16"):
        local_rank_to_print = local_rank
    else:
        local_rank_to_print = 0

    if torch.cuda.is_available():
        if(local_rank_to_print == 0):
            print("=================================================================", flush = True)
            print("==================== PARAMETER CONFIGURATION ====================", flush = True)
            print("=================================================================", flush = True)
            print(f"model={model_name}", flush=True)
            print(f"criterion={criterion_name}", flush=True)
            print(f"C={C}", flush=True)
            print(f"delta={args.delta}", flush=True)
            print(f"lr={lr}", flush=True)    
            print(f"lambda_reg={lambda_reg}", flush=True)
            print(f"alpha={alpha}", flush=True)    
            print(f"[T1=lambda_reg*alpha={round(lambda_reg*alpha, 6)}]", flush=True)
            print(f"[T2=lambda_reg*(1-alpha)={round(lambda_reg*(1-alpha), 6)}]", flush=True)
            print(f"subgradient_step={subgradient_step}", flush=True)    
            print(f"w0={w0}", flush=True)    
            print(f"r={r}", flush=True)  
            print(f"bucket_zero={bucket_zero}", flush=True)  
            print(f"BestQuantization_target_acc={BestQuantization_target_acc}", flush=True)    
            print(f"final_target_acc={final_target_acc}", flush=True)
            print(f"target_zstd_ratio={target_zstd_ratio}", flush=True)    
            print(f"min_xi={min_xi}", flush=True)    
            print(f"max_xi={max_xi}", flush=True)  
            print(f"upper_c={upper_c}", flush=True)
            print(f"lower_c={lower_c}", flush=True)  
            print(f"c1={c1}", flush=True)
            print(f"c2={c2}", flush=True)
            print(f"first_best_indices={first_best_indices}", flush=True)
            print(f"accuracy_tollerance={accuracy_tollerance}", flush=True)
            print(f"zeta={zeta}", flush=True)
            print(f"l={l}", flush=True)
            print(f"n_epochs={n_epochs}", flush=True) 
            print(f"max_iterations={max_iterations}", flush=True)    
            print(f"train_optimizer={train_optimizer}", flush=True)    
            print(f"entropy_optimizer={entropy_optimizer}", flush=True) 
            print(f"pruning={pruning}", flush=True)
            print(f"QuantizationType={QuantizationType}", flush=True)
            print(f"sparsity_threshold={sparsity_threshold}", flush=True)
            print("-"*60, flush=True)       
    else:
        if(args.delta == 5): 
            print("=================================================================", flush = True)
            print("==================== PARAMETER CONFIGURATION ====================", flush = True)
            print("=================================================================", flush = True)
            print(f"model={model_name}", flush=True)
            print(f"criterion={criterion_name}", flush=True)
            print(f"C={C}", flush=True)
            print(f"delta={args.delta}", flush=True)
            print(f"lr={lr}", flush=True)    
            print(f"lambda_reg={lambda_reg}", flush=True)
            print(f"alpha={alpha}", flush=True)    
            print(f"[T1=lambda_reg*alpha={round(lambda_reg*alpha, 6)}]", flush=True)
            print(f"[T2=lambda_reg*(1-alpha)={round(lambda_reg*(1-alpha), 6)}]", flush=True)
            print(f"subgradient_step={subgradient_step}", flush=True)    
            print(f"w0={w0}", flush=True)    
            print(f"r={r}", flush=True)  
            print(f"bucket_zero={bucket_zero}", flush=True)  
            print(f"BestQuantization_target_acc={BestQuantization_target_acc}", flush=True)    
            print(f"final_target_acc={final_target_acc}", flush=True)
            print(f"target_zstd_ratio={target_zstd_ratio}", flush=True)    
            print(f"min_xi={min_xi}", flush=True)    
            print(f"max_xi={max_xi}", flush=True)  
            print(f"upper_c={upper_c}", flush=True)
            print(f"lower_c={lower_c}", flush=True)  
            print(f"c1={c1}", flush=True)
            print(f"c2={c2}", flush=True)
            print(f"first_best_indices={first_best_indices}", flush=True)
            print(f"accuracy_tollerance={accuracy_tollerance}", flush=True)
            print(f"zeta={zeta}", flush=True)
            print(f"l={l}", flush=True)
            print(f"n_epochs={n_epochs}", flush=True) 
            print(f"max_iterations={max_iterations}", flush=True)    
            print(f"train_optimizer={train_optimizer}", flush=True)    
            print(f"entropy_optimizer={entropy_optimizer}", flush=True) 
            print(f"pruning={pruning}", flush=True)
            print(f"QuantizationType={QuantizationType}", flush=True)
            print(f"sparsity_threshold={sparsity_threshold}", flush=True)
            print("-"*60, flush=True)   
            print("Using CPU.", flush=True)          

    # Load the training and test datasets using the load_data function
    if(model_name == "AlexNet" or model_name == "VGG16"):
        trainset, testset, train_sampler = load_data(model_name)
        trainloader = trainset
        testloader = testset
        train_and_evaluate(
            model=model, model_name=model_name, criterion=criterion, C=C, lr=lr, lambda_reg=lambda_reg, alpha=alpha,
            subgradient_step=subgradient_step, w0=w0, r=r, first_best_indices=first_best_indices,
            BestQuantization_target_acc=BestQuantization_target_acc, final_target_acc=final_target_acc, 
            target_zstd_ratio=target_zstd_ratio, min_xi=min_xi, max_xi=max_xi, upper_c=upper_c, lower_c=lower_c, c1=c1, c2=c2, 
            zeta=zeta, l=l, n_epochs=n_epochs, max_iterations=max_iterations, device=device, train_optimizer=train_optimizer,
            entropy_optimizer=entropy_optimizer, trainloader=trainloader, testloader=testloader, train_sampler=train_sampler,
            delta=args.delta, pruning=pruning, QuantizationType=QuantizationType, sparsity_threshold=sparsity_threshold, 
            accuracy_tollerance=accuracy_tollerance
        )   
        cleanup()     
    else:
        trainset, testset = load_data(model_name)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, drop_last=True, num_workers=0)
        testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False, num_workers=0)
        train_and_evaluate(
            model=model, model_name=model_name, criterion=criterion, C=C, lr=lr, lambda_reg=lambda_reg, alpha=alpha,
            subgradient_step=subgradient_step, w0=w0, r=r, first_best_indices=first_best_indices,
            BestQuantization_target_acc=BestQuantization_target_acc, final_target_acc=final_target_acc, 
            target_zstd_ratio=target_zstd_ratio, min_xi=min_xi, max_xi=max_xi, upper_c=upper_c, lower_c=lower_c, c1=c1, c2=c2, 
            zeta=zeta, l=l, n_epochs=n_epochs, max_iterations=max_iterations, device=device, train_optimizer=train_optimizer,
            entropy_optimizer=entropy_optimizer, trainloader=trainloader, testloader=testloader, train_sampler=None,
            delta=args.delta, pruning=pruning, QuantizationType=QuantizationType, sparsity_threshold=sparsity_threshold, 
            accuracy_tollerance=accuracy_tollerance
        )
