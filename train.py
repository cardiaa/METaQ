import torch  
import time
import numpy as np
import os
from torchvision import datasets, transforms  
from itertools import product
from utils.trainer import train_and_evaluate  
import multiprocessing


def set_affinity(process_index, num_processes):
    num_total_cores = os.cpu_count()
    cores_per_process = 8  # Numero di core che ogni processo deve utilizzare
    required_cores = num_processes * cores_per_process

    if required_cores > num_total_cores:
        raise ValueError(f"Non ci sono abbastanza core per assegnare {cores_per_process} core per processo. "
                         f"Richiesti: {required_cores}, Disponibili: {num_total_cores}")

    # Determina quali core assegnare a questo processo
    start_index = process_index * cores_per_process
    core_indices = list(range(start_index, start_index + cores_per_process))
    
    # Imposta l'affinità per il processo
    os.sched_setaffinity(0, set(core_indices))

    print(f"Processo {process_index}: Affinità impostata su core {core_indices}", flush=True)




def load_data():
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    return trainset, testset


def train_model(args):

    process_index = args[-2]  # Penultimo argomento è l'indice del processo
    num_processes = args[-1]  # Ultimo argomento è il numero totale di processi

    set_affinity(process_index, num_processes)  # Commentata per ora

    torch.set_num_threads(8)
    
    
    #print(f"Process {process_index}: torch.get_num_threads() = {torch.get_num_threads()}")
    #print(f"Process {process_index}: Affinity = {os.sched_getaffinity(0)}", flush=True)

    trainset, testset = load_data()  # Carichiamo i dati localmente
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False, num_workers=0)

    (C, lr, lambda_reg, alpha, subgradient_step, w0, r,
     target_acc, target_entr, min_xi, max_xi, n_epochs,
     device, train_optimizer, entropy_optimizer) = args[:-2]

    print(f"Process {process_index}: Dati caricati", flush=True)

    start_time = time.time()

    accuracy, entropy, target_acc, target_entr = train_and_evaluate(
        C=C, lr=lr, lambda_reg=lambda_reg, alpha=alpha, subgradient_step=subgradient_step,
        w0=w0, r=r, target_acc=target_acc, target_entr=target_entr,
        min_xi=min_xi, max_xi=max_xi, n_epochs=n_epochs,
        device=device, train_optimizer=train_optimizer,
        entropy_optimizer=entropy_optimizer,
        trainloader=trainloader, testloader=testloader
    )

    training_time = time.time() - start_time

    print(f"Process {process_index}: Training completato in {training_time:.2f} secondi", flush=True)

    return (C, r, training_time)


if __name__ == "__main__":
    num_processes = 12  
    num_total_cores = os.cpu_count()
    cores_per_process = 8

    if num_processes * cores_per_process > num_total_cores:
        raise ValueError(f"Il numero di core disponibili non è sufficiente per garantire {cores_per_process} core per processo.")

    print(f"Numero di processi: {num_processes}")
    print(f"Numero totale di core logici disponibili: {num_total_cores}")

    multiprocessing.set_start_method('spawn', force=True)
    device = torch.device("cpu")
    np.set_printoptions(precision=6)

    # Caricamento dei dati globale per ridurre l'I/O
    global_trainloader, global_testloader = load_data()

    param_grid = {
        "C": [6],
        "lr": [0.0007],
        "lambda_reg": [0.0015],
        "alpha": [0.533],
        "subgradient_step": [1e5],
        "w0": [-0.11],
        "r": [round(1.1 + i * 0.002, 3) for i in range(num_processes)],
        "target_acc": [98.99],
        "target_entr": [0.99602e6],
        "min_xi": [0],
        "max_xi": [1],
        "n_epochs": [100],
        "device": [device],
        "train_optimizer": ['A'],
        "entropy_optimizer": ['F'],
    }

    param_combinations = [(params + (i, num_processes)) for i, params in enumerate(product(
        param_grid["C"], param_grid["lr"], param_grid["lambda_reg"],
        param_grid["alpha"], param_grid["subgradient_step"], param_grid["w0"],
        param_grid["r"], param_grid["target_acc"], param_grid["target_entr"],
        param_grid["min_xi"], param_grid["max_xi"], param_grid["n_epochs"],
        param_grid["device"], param_grid["train_optimizer"],
        param_grid["entropy_optimizer"]
    ))]

    with multiprocessing.Pool(processes=num_processes, maxtasksperchild=1) as pool:
        results = pool.map(train_model, param_combinations)
    
    print("Tutti i processi completati.")

