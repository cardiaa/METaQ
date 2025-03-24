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
    
    core_indices = [i for i in range(num_total_cores) if i % num_processes == process_index]
    os.sched_setaffinity(0, core_indices)


def load_data():
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    return trainset, testset


def train_model(args):

    process_index = args[-6]  # Indice del processo
    num_processes = args[-5]  # Numero di processi
    datasets = args[-4]  # Tuple di dataset
    arrival_times = args[-3]  # Lista dei tempi di arrivo
    sync_failed = args[-2]  # Variabile di sincronizzazione
    sync_lock = args[-1]  # Lock di sincronizzazione

    set_affinity(process_index, num_processes)  
    torch.set_num_threads(1)

    trainset, testset = datasets
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False, num_workers=0)

    (C, lr, lambda_reg, alpha, subgradient_step, w0, r,
     target_acc, target_entr, min_xi, max_xi, n_epochs,
     device, train_optimizer, entropy_optimizer) = args[:-5]

    print(f"Process {process_index}: Dati caricati", flush=True)
    start_time = time.time()

    accuracy, entropy, target_acc, target_entr = train_and_evaluate(
        C=C, lr=lr, lambda_reg=lambda_reg, alpha=alpha, subgradient_step=subgradient_step,
        w0=w0, r=r, target_acc=target_acc, target_entr=target_entr,
        min_xi=min_xi, max_xi=max_xi, n_epochs=n_epochs,
        device=device, train_optimizer=train_optimizer,
        entropy_optimizer=entropy_optimizer,
        trainloader=trainloader, testloader=testloader,
        process_index=process_index, num_processes=num_processes, 
        arrival_times=arrival_times, sync_lock=sync_lock, sync_failed=sync_failed
    )


    # Rilevamento dei tempi di arrivo
    with arrival_times.get_lock():
        arrival_times[process_index] = time.time()
    
    time.sleep(0.1)  # Ritardo per garantire che tutti i processi scrivano il proprio timestamp

    with arrival_times.get_lock():
        first_arrival = min(arrival_times)
        last_arrival = max(arrival_times)
        difference = last_arrival - first_arrival

        if difference > 0.5:
            print(f"Processi non sincronizzati: {difference:.2f} secondi di differenza")
            sync_failed.value = True  # Notifica al main() che la sincronizzazione è fallita
        else:
            print(f"Sincronizzazione riuscita con differenza di {difference:.2f} secondi")

    training_time = time.time() - start_time
    print(f"Process {process_index}: Training completato in {training_time:.2f} secondi", flush=True)
    return (C, r, training_time)


if __name__ == "__main__":
    num_processes = 12  
    num_total_cores = os.cpu_count()  

    print(f"Numero di processi: {num_processes}")
    print(f"Numero totale di core logici disponibili: {num_total_cores}")

    multiprocessing.set_start_method('spawn', force=True)
    device = torch.device("cpu")
    print(device)
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

    param_combinations = [(params + (i, num_processes, (trainset, testset)))
                          for i, params in enumerate(product(
                              param_grid["C"], param_grid["lr"], param_grid["lambda_reg"],
                              param_grid["alpha"], param_grid["subgradient_step"], param_grid["w0"],
                              param_grid["r"], param_grid["target_acc"], param_grid["target_entr"],
                              param_grid["min_xi"], param_grid["max_xi"], param_grid["n_epochs"],
                              param_grid["device"], param_grid["train_optimizer"],
                              param_grid["entropy_optimizer"]
                          ))]

    while True:  
        with multiprocessing.Manager() as manager:
            arrival_times = manager.list([-1] * num_processes) 
            sync_failed = manager.Value('b', False)
            sync_lock = manager.Lock()  # Aggiungo il lock
            
            enhanced_combinations = [params + (arrival_times, sync_failed, sync_lock) for params in param_combinations]
            
            pool = multiprocessing.Pool(processes=num_processes, maxtasksperchild=1)
            try:
                results = pool.map(train_model, enhanced_combinations)
            except Exception as e:
                print(f"Errore durante l'esecuzione del pool: {e}")
                pool.terminate()
                pool.join()
                continue
            
            pool.close()
            pool.join()
            
            if sync_failed.value:
                print("❌ Riprova: I processi non sono stati sincronizzati correttamente. Rilancio di tutti i processi...\n")
            else:
                print("✅ Tutti i processi completati e sincronizzati correttamente.")
                break
