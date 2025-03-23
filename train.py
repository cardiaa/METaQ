import torch
import time
import numpy as np
import os
from torchvision import datasets, transforms
from itertools import product
from utils.trainer import train_and_evaluate
import multiprocessing
import threading


def set_affinity(process_index, num_processes):
    num_total_cores = os.cpu_count()
    cores_per_process = max(1, num_total_cores // num_processes)  

    # Distribuzione più distanziata dei core
    core_indices = [i for i in range(num_total_cores) if i % num_processes == process_index]
    os.sched_setaffinity(0, core_indices)


# Variabile globale per sincronizzazione
reached_ten_event = threading.Event()


def load_data():
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    return trainset, testset


def train_model(args):

    process_index = args[-3]
    num_processes = args[-2]
    datasets = args[-1]

    set_affinity(process_index, num_processes)
    torch.set_num_threads(1)

    trainset, testset = datasets
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False, num_workers=0)

    (C, lr, lambda_reg, alpha, subgradient_step, w0, r,
     target_acc, target_entr, min_xi, max_xi, n_epochs,
     device, train_optimizer, entropy_optimizer) = args[:-3]

    print(f"Process {process_index}: Dati caricati", flush=True)

    for i, data in enumerate(trainloader, 0):
        if i == 10:  # Punto di sincronizzazione
            reached_ten_event.set()  # Notifica che questo processo è arrivato a 10
            print(f"Sono arrivato a 10. - Process {process_index}", flush=True)
        if reached_ten_event.is_set():
            break

    start_time = time.time()

    # Richiamo alla funzione di allenamento
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


def main():
    global reached_ten_event

    num_processes = 12  
    num_total_cores = os.cpu_count()  

    print(f"Numero di processi: {num_processes}")
    print(f"Numero totale di core logici disponibili: {num_total_cores}")

    multiprocessing.set_start_method('spawn', force=True)
    device = torch.device("cpu")
    np.set_printoptions(precision=6)

    trainset, testset = load_data()

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

    param_combinations = [(params + (i, num_processes, (trainset, testset))) for i, params in enumerate(product(
        param_grid["C"], param_grid["lr"], param_grid["lambda_reg"],
        param_grid["alpha"], param_grid["subgradient_step"], param_grid["w0"],
        param_grid["r"], param_grid["target_acc"], param_grid["target_entr"],
        param_grid["min_xi"], param_grid["max_xi"], param_grid["n_epochs"],
        param_grid["device"], param_grid["train_optimizer"],
        param_grid["entropy_optimizer"]
    ))]

    while True:
        reached_ten_event.clear()
        with multiprocessing.Pool(processes=num_processes, maxtasksperchild=1) as pool:
            results = pool.map(train_model, param_combinations)

        time.sleep(0.1)

        if reached_ten_event.is_set():
            print("Tutti i processi hanno raggiunto l'indice 10 correttamente.")
            break
        else:
            print("Ritardo rilevato, riavvio dei processi...")

    print("Tutti i processi completati.")


if __name__ == "__main__":
    main()
