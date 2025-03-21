import os
import torch  
import time
import numpy as np
from torchvision import datasets, transforms  
from itertools import product
from utils.trainer import train_and_evaluate  
import multiprocessing

def set_affinity(process_index):
    """ Imposta l'affinità del processo sui core fisici (escludendo i core logici) """
    # Lista di core fisici (escludendo i core logici) - per esempio, su un sistema con 192 core fisici
    core_list = list(range(0, 192, 2))  # Core fisici, escludendo i logici
    os.sched_setaffinity(0, core_list)  # Associa il processo corrente ai core fisici

def load_data():
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    return trainset, testset

def train_model(args):
    process_index = args[-2]  # Penultimo argomento è l'indice del processo
    num_processes = args[-1]  # Ultimo argomento è il numero totale di processi
    
    # Imposta l'affinità del processo sui core fisici
    set_affinity(process_index)
    
    # Limita il numero di thread di PyTorch
    os.environ["OMP_NUM_THREADS"] = "1"  # Limita il numero di thread a 1
    os.environ["MKL_NUM_THREADS"] = "1"  # Limita il numero di thread MKL a 1
    torch.set_num_threads(1)

    # Caricamento dei dati
    trainset, testset = load_data()
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False, num_workers=0)

    # Estrazione dei parametri per l'addestramento
    (C, lr, lambda_reg, alpha, subgradient_step, w0, r,
     target_acc, target_entr, min_xi, max_xi, n_epochs,
     device, train_optimizer, entropy_optimizer) = args[:-2]

    print(f"Process {process_index}: Dati caricati", flush=True)

    # Tempo di inizio addestramento
    start_time = time.time()

    # Esegui l'addestramento e la valutazione
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
    num_processes = 20  # Imposta il numero di processi desiderato
    num_total_cores = os.cpu_count()  # Numero di core logici disponibili

    print(f"Numero di processi: {num_processes}")
    print(f"Numero totale di core logici disponibili: {num_total_cores}")

    # Imposta il metodo di avvio dei processi (esempio su Unix-like)
    multiprocessing.set_start_method('spawn', force=True)
    
    # Definisci il dispositivo (GPU se disponibile)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    np.set_printoptions(precision=6)

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

    # Combinazioni di parametri per ogni processo
    param_combinations = [(params + (i, num_processes)) for i, params in enumerate(product(
        param_grid["C"], param_grid["lr"], param_grid["lambda_reg"],
        param_grid["alpha"], param_grid["subgradient_step"], param_grid["w0"],
        param_grid["r"], param_grid["target_acc"], param_grid["target_entr"],
        param_grid["min_xi"], param_grid["max_xi"], param_grid["n_epochs"],
        param_grid["device"], param_grid["train_optimizer"],
        param_grid["entropy_optimizer"]
    ))]

    # Crea i processi
    processes = []

    for i in range(num_processes):
        p = multiprocessing.Process(target=train_model, args=(param_combinations[i],))  # Passa l'argomento giusto
        processes.append(p)
        p.start()  # Avvia il processo

    # Aspetta che ogni processo finisca
    for p in processes:
        p.join()

    print("Tutti i processi completati.")
