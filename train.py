import torch  
import time
import numpy as np
import os
from torchvision import datasets, transforms  
from itertools import product
from utils.trainer import train_and_evaluate  
import multiprocessing
from datetime import datetime


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


process_start_times = multiprocessing.Manager().dict()


def train_model(args):

    process_index = args[-3]  # Terzultimo argomento è l'indice del processo
    num_processes = args[-2]  # Penultimo argomento è il numero totale di processi
    datasets = args[-1]  # Ultimo argomento è il tuple (trainset, testset)

    set_affinity(process_index, num_processes)  

    torch.set_num_threads(1)

    trainset, testset = datasets  
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False, num_workers=0)

    start_time = time.time()
    process_start_times[process_index] = start_time

    # Attendi tutti i processi prima di procedere
    while len(process_start_times.keys()) < num_processes:
        time.sleep(0.01)

    print(f"Process {process_index}: Dati caricati", flush=True)

    # Sincronizzazione per il controllo "Sono arrivato a 10."
    for i, data in enumerate(trainloader, 0):
        if i == 10:
            process_start_times[process_index] = time.time()
            break

    # Aspetta che tutti i processi raggiungano l'indice 10
    while len(process_start_times.keys()) < num_processes:
        time.sleep(0.01)

    timestamps = list(process_start_times.values())
    timestamps.sort()
    if timestamps[-1] - timestamps[0] > 0.1:
        print("I processi non sono sincronizzati! Riavvio il programma...", flush=True)
        os._exit(1)  # Terminazione immediata dell'intero programma

    print(f"Sono arrivato a 10.", flush=True)
    return process_index  # Restituisci l'indice del processo per completare correttamente il pool


if __name__ == "__main__":
    num_processes = 12  

    print(f"Numero di processi: {num_processes}")
    print(f"Numero totale di core logici disponibili: {os.cpu_count()}")

    multiprocessing.set_start_method('spawn', force=True)

    trainset, testset = load_data()  

    param_combinations = [(0, 0.0007, 0.0015, 0.533, 1e5, -0.11,
                           round(1.1 + i * 0.002, 3), 98.99, 0.99602e6,
                           0, 1, 100, torch.device("cpu"), 'A', 'F',
                           i, num_processes, (trainset, testset))
                          for i in range(num_processes)]

    while True:  # Continua a riavviare finché non è sincronizzato
        process_start_times = multiprocessing.Manager().dict()

        with multiprocessing.Pool(processes=num_processes, maxtasksperchild=1) as pool:
            results = pool.map(train_model, param_combinations)

        if len(results) == num_processes:  # Tutti i processi sono stati completati correttamente
            break  

    print("Tutti i processi completati correttamente.")
