import torch  
import time
import numpy as np
import os
from torchvision import datasets, transforms  
from itertools import product
from utils.trainer import train_and_evaluate  
from concurrent.futures import ProcessPoolExecutor


def set_affinity(process_index, num_processes):
    num_total_cores = os.cpu_count()
    cores_per_process = max(1, num_total_cores // num_processes)  
    start_core = process_index * cores_per_process
    end_core = start_core + cores_per_process
    os.sched_setaffinity(0, range(start_core, min(end_core, num_total_cores)))


def load_data():
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    return trainset, testset


def train_model(args):
    process_index = args[-2]  # Penultimo argomento è l'indice del processo
    num_processes = args[-1]  # Ultimo argomento è il numero totale di processi

    #set_affinity(process_index, num_processes)  # Commentata per ora

    torch.set_num_threads(1)
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


def train_process(args):
    """ Funzione separata per evitare l'errore di pickle su funzioni locali. """
    return train_model(args)


def train_in_batches(param_combinations, batch_size):
    """
    Funzione che divide i processi in batch più piccoli.
    """
    for i in range(0, len(param_combinations), batch_size):
        batch = param_combinations[i:i + batch_size]
        with ProcessPoolExecutor(max_workers=batch_size) as executor:
            # Ora chiamiamo direttamente la funzione train_process
            results = list(executor.map(train_process, batch))
            print(f"Batch {i // batch_size + 1} completato")
        time.sleep(2)  # Una piccola pausa per evitare sovraccarichi


if __name__ == "__main__":
    num_processes = 12  # Imposta il numero di processi desiderato
    num_total_cores = os.cpu_count()  

    print(f"Numero di processi: {num_processes}")
    print(f"Numero totale di core logici disponibili: {num_total_cores}")

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

    param_combinations = [(params + (i, num_processes)) for i, params in enumerate(product(
        param_grid["C"], param_grid["lr"], param_grid["lambda_reg"],
        param_grid["alpha"], param_grid["subgradient_step"], param_grid["w0"],
        param_grid["r"], param_grid["target_acc"], param_grid["target_entr"],
        param_grid["min_xi"], param_grid["max_xi"], param_grid["n_epochs"],
        param_grid["device"], param_grid["train_optimizer"],
        param_grid["entropy_optimizer"]
    ))]

    # Avvia i processi in batch
    train_in_batches(param_combinations, batch_size=6)  # Esegui in batch più piccoli

    print("Tutti i processi completati.")
