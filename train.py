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
    cores_per_process = max(1, num_total_cores // num_processes)  
    
    # Distribuzione più distanziata dei core
    core_indices = [i for i in range(num_total_cores) if i % num_processes == process_index]
    os.sched_setaffinity(0, core_indices)


def load_data():
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    return trainset, testset


def train_model(args):
    process_index = args[-2]  # Penultimo argomento è l'indice del processo
    num_processes = args[-1]  # Ultimo argomento è il numero totale di processi

    set_affinity(process_index, num_processes)  # Commentata per ora

    torch.set_num_threads(1)
    
    
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
        entropy_optimizer=entropy_optimizer
    )

    training_time = time.time() - start_time

    print(f"Process {process_index}: Training completato in {training_time:.2f} secondi", flush=True)

    return (C, r, training_time)


def worker(semaphore, args):
    # Ogni processo segnala di essere partito
    semaphore.release()
    return train_model(args)


def run_in_parallel(param_combinations, num_processes, max_wait_time=0.5):
    semaphore = multiprocessing.Semaphore(0)  # Semaforo inizializzato a 0

    with multiprocessing.Pool(processes=num_processes) as pool:
        # Avviamo tutti i processi asincroni
        print("aaaa")
        async_results = [pool.apply_async(worker, (semaphore, param)) for param in param_combinations]
        print("bbbb")
        # Controlliamo che tutti i processi siano partiti entro max_wait_time
        start_time = time.time()
        for _ in param_combinations:
            # Aspetta che ogni processo parta
            semaphore.acquire()
        
        elapsed_time = time.time() - start_time
        if elapsed_time > max_wait_time:
            print(f"Attenzione! Non tutti i processi sono partiti in {max_wait_time} secondi. Tempo trascorso: {elapsed_time:.2f} secondi.")
        else:
            print(f"Tutti i processi sono partiti in {elapsed_time:.2f} secondi.")

        # Attendere il completamento dei risultati
        results = [result.get() for result in async_results]
    
    return results


if __name__ == "__main__":
    num_processes = 12  # Numero desiderato di processi
    num_total_cores = os.cpu_count()  

    print(f"Numero di processi: {num_processes}")
    print(f"Numero totale di core logici disponibili: {num_total_cores}")

    multiprocessing.set_start_method('spawn', force=True)
    device = torch.device("cpu")
    print(device)
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

    # Avvia i processi in parallelo
    results = run_in_parallel(param_combinations, num_processes)
    
    print("Tutti i processi completati.")
