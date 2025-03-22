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

    set_affinity(process_index, num_processes)  # Attivo affinità sui core

    torch.set_num_threads(1)  # Limito ogni processo a un singolo thread
    
    # Stampa per verifica dell'affinità
    print(f"Process {process_index}: torch.get_num_threads() = {torch.get_num_threads()}")
    print(f"Process {process_index}: Affinity = {os.sched_getaffinity(0)}", flush=True)
    
    trainset, testset = load_data()  # Caricamento locale dei dati
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False, num_workers=0)

    (C, lr, lambda_reg, alpha, subgradient_step, w0, r,
     target_acc, target_entr, min_xi, max_xi, n_epochs,
     device, train_optimizer, entropy_optimizer) = args[:-2]

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
    C_list = [0.1, 1, 10]
    r_list = [0.5, 1.0]

    num_processes = min(len(C_list) * len(r_list), os.cpu_count())

    combinations = list(product(C_list, r_list))

    pool = multiprocessing.Pool(processes=num_processes)

    results = pool.map(train_model, [(C, 0.001, 0.01, 0.1, 0.01, 0, r, 0.95, 0.01, 0, 1, 5, 'cpu', 'SGD', 'Adam', idx, num_processes) 
                                     for idx, (C, r) in enumerate(combinations)])

    pool.close()
    pool.join()

    print("\nRisultati finali:")
    for result in results:
        print(f"C: {result[0]}, r: {result[1]}, Tempo di allenamento: {result[2]:.2f} secondi")
