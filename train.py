import argparse
import torch
import os
from utils.trainer import train_and_evaluate
from torchvision import datasets, transforms

def load_data():
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    return trainset, testset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--delta", type=float, required=True, help="Valore di delta")
    args = parser.parse_args()

    # Limitare l'utilizzo dei thread a 1
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"

    trainset, testset = load_data()
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False, num_workers=0)

    # Definizione dei parametri fissi
    C = 6
    lr = 0.0007
    lambda_reg = 0.0015
    alpha = 0.533
    subgradient_step = 1e5
    w0 = -0.11
    r = 1.1
    target_acc = 98.99
    target_entr = 0.99602e6
    min_xi = 0
    max_xi = 1
    n_epochs = 100
    device = torch.device("cpu")
    train_optimizer = "A"
    entropy_optimizer = "F"

    # Allenamento del modello con il valore di r passato da terminale
    print(f"Avvio training con delta = {args.delta}")
    train_and_evaluate(
        C=C, lr=lr, lambda_reg=lambda_reg, alpha=alpha,
        subgradient_step=subgradient_step, w0=w0, r=r,
        target_acc=target_acc, target_entr=target_entr,
        min_xi=min_xi, max_xi=max_xi, n_epochs=n_epochs,
        device=device, train_optimizer=train_optimizer,
        entropy_optimizer=entropy_optimizer,
        trainloader=trainloader, testloader=testloader, 
        delta=args.delta
    )
    