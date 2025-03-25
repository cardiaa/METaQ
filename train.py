import torch  
import time
import numpy as np
import os
import argparse
from torchvision import datasets, transforms  
from utils.trainer import train_and_evaluate


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

    # Estrai il valore di r da args
    r = args.r

    trainset, testset = load_data()  
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False, num_workers=0)

    # Altri parametri (rimosso per brevità)
    C = 6
    lr = 0.0007
    lambda_reg = 0.0015
    alpha = 0.533
    subgradient_step = 1e5
    w0 = -0.11
    target_acc = 98.99
    target_entr = 0.99602e6
    min_xi = 0
    max_xi = 1
    n_epochs = 100
    device = torch.device("cpu")
    train_optimizer = 'A'
    entropy_optimizer = 'F'

    print(f"Training started for r = {r}")

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

    print(f"Training completed for r = {r} in {training_time:.2f} seconds")


if __name__ == "__main__":
    # Aggiungi l'argomento per `r` tramite argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--r', type=float, required=True, help='Valore di r da passare al modello')
    args = parser.parse_args()

    # Chiamata alla funzione di allenamento con il parametro r
    train_model(args)
