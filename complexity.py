import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from collections import Counter
from utils.trainer import train_and_evaluate
from utils.knapsack import knapsack_specialized_single

if __name__ == "__main__":
    # Initializes the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fixed parameters
    n_istances = 10000 # Number of simulations
    C = 256 # Number of buckets of quantization
    N = 1 # N = 1 means no parallelization
    v = torch.linspace(0, (C - 1)/C, C).to(device) # Quantization vector

    # Memorizes the number of breakpoints for each istance of the problem
    iterations = []

    for i in range(n_istances):

        # Initializes variables
        xi = torch.sort(torch.rand(C, dtype=torch.float32)).values
        w = v[torch.randint(0, C, (N,), device=device)]

        # Solves the optimization problem
        x1, lambda1, optimal_value1, conta_iterazioni = knapsack_specialized_single(xi.tolist(), v.tolist(), float(w))
        iterations.append(conta_iterazioni)

    # Counts occurencies of the elements in the list
    occorrenze = Counter(iterations)

    # Extracts keys and values for the histogram
    elementi = list(occorrenze.keys())
    frequenze = list(occorrenze.values())

    # Creates histogram
    plt.bar(elementi, frequenze, color='skyblue', edgecolor='black')
    plt.xlabel('Number of Breakpoints')
    plt.ylabel('Frequency')
    plt.title(f'Occurencies histogram (C = {C})')
    plt.show()
