import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from collections import Counter
from utils.trainer import train_and_evaluate
from utils.knapsack import knapsack_specialized_histo

if __name__ == "__main__":
    # Select device based on availability of CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    C = 256
    M = 1

    iterations = []

    # Run the algorithm 10,000 times
    for i in range(10000):
        xi = torch.sort(torch.rand(C, device=device))[0]  
        v = torch.linspace(0, 1 - (1 / C), C, device=device)  
        #w = v[torch.randint(0, C, (1,), device=device)]  
        w = torch.rand(M, device=device)

        x_opt, lambda_opt, optimal_value, iterations_count = knapsack_specialized_histo(xi, v, w, C)

        iterations.append(iterations_count)

    # Count occurrences of elements in the list
    occurrences = Counter(iterations)

    # Extract keys and values for histogram
    elements = list(occurrences.keys())
    frequencies = list(occurrences.values())

    # Create histogram
    plt.bar(elements, frequencies, color="skyblue", edgecolor="black")
    plt.xlabel("# of Breakpoints")
    plt.ylabel("Frequency")
    plt.title("Histogram of occurrences (C = 256)")
    plt.show()