import torch  
import time
import numpy as np
import os
from torchvision import datasets, transforms  
from itertools import product
from utils.trainer import train_and_evaluate  
import multiprocessing

def train_model(args):

    torch.set_num_threads(1)

    # Unpack arguments
    (C, lr, lambda_reg, alpha, subgradient_step, w0, r,
     target_acc, target_entr, min_xi, max_xi, n_epochs,
     device, train_optimizer, entropy_optimizer) = args

    # Crea i DataLoader all'interno del processo figlio
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0)
    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False, num_workers=0)
    
    print("Dati caricati", flush=True)

    # Inizia il training
    start_time = time.time()
        
    # Creiamo un file di log per ogni combinazione
    #output_dir = "training_logs"
    #os.makedirs(output_dir, exist_ok=True)
    #log_filename = f"{output_dir}/log_C_{C}_r_{r}_proc_{os.getpid()}.txt"
    
    #with open(log_filename, "w") as f:
    #    f.write(f"C={C}, lr={lr}, lambda_reg={lambda_reg}, "
    #        f"alpha={alpha}, subgradient_step={subgradient_step}, w0={w0}, r={r}, "
    #        f"target_acc={target_acc}, target_entr={target_entr}, "
    #        f"min_xi={min_xi}, max_xi={max_xi}, n_epochs={n_epochs}, train_optimizer={train_optimizer} "
    #        f"entropy_optimizer={entropy_optimizer}")
    
    accuracy, entropy, target_acc, target_entr = train_and_evaluate(
        C=C, lr=lr, lambda_reg=lambda_reg, alpha=alpha, subgradient_step=subgradient_step,
        w0=w0, r=r, target_acc=target_acc, target_entr=target_entr,
        min_xi=min_xi, max_xi=max_xi, n_epochs=n_epochs,
        device=device, train_optimizer=train_optimizer,
        entropy_optimizer=entropy_optimizer,
        trainloader=trainloader, testloader=testloader
    )
    
    training_time = time.time() - start_time

    #with open(log_filename, "a") as f:
    #    f.write(f"Training completed for C={C}, r={r}\n")
    #    f.write(f"Accuracy: {accuracy}\n")
    #    f.write(f"Entropy: {entropy}\n")
    #    f.write(f"Target Accuracy: {target_acc}\n")
    #    f.write(f"Target Entropy: {target_entr}\n")
    #    f.write(f"Training Time: {training_time:.2f} seconds\n")

    return (C, r, training_time)



if __name__ == "__main__":
    num_processes = 20
    num_total_cores = os.cpu_count() # Numero totale di core disponibili
    torch_threads_per_process = max(1, num_total_cores // num_processes)  # Thread per processo

    print(f"Numero di processi: {num_processes}")
    print(f"Thread per processo: {torch_threads_per_process}")

    multiprocessing.set_start_method('spawn', force=True)
    print(f"Thread set for PyTorch: {torch.get_num_threads()}")
    print(f"Number of core available on the machine: {num_total_cores}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    np.set_printoptions(precision=6)

    param_grid = {
        "C": [6],
        "lr": [0.0007],
        "lambda_reg": [0.0015],
        "alpha": [0.533],
        "subgradient_step": [1e5],
        "w0": [-0.11],
        "r": [round(1.1 + i * 0.002, 3) for i in range(20)],
        "target_acc": [98.99],
        "target_entr": [0.99602e6],
        "min_xi": [0],
        "max_xi": [1],
        "n_epochs": [100],
        "device": [device],
        "train_optimizer": ['A'],
        "entropy_optimizer": ['F'],
    }

    # Creazione della lista di combinazioni di parametri
    param_combinations = list(product(
        param_grid["C"], param_grid["lr"], param_grid["lambda_reg"],
        param_grid["alpha"], param_grid["subgradient_step"], param_grid["w0"],
        param_grid["r"], param_grid["target_acc"], param_grid["target_entr"],
        param_grid["min_xi"], param_grid["max_xi"], param_grid["n_epochs"],
        param_grid["device"], param_grid["train_optimizer"],
        param_grid["entropy_optimizer"]
    ))


    # Numero di processi da lanciare in parallelo
    num_processes = min(384, len(param_combinations))  # Non ha senso lanciare più processi delle combinazioni

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(train_model, param_combinations)
