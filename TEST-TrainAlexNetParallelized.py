import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms, models

def setup(rank, world_size):
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size, delta):
    print(f"[GPU {rank}] starting training with delta={delta}")
    setup(rank, world_size)

    # Transform
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor()
    ])

    # Dataset and sampler
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=64, sampler=sampler)

    # Model
    model = models.alexnet()
    model = model.cuda(rank)
    model = DDP(model, device_ids=[rank])

    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss().cuda(rank)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Training loop
    for epoch in range(delta):
        sampler.set_epoch(epoch)
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.cuda(rank, non_blocking=True)
            targets = targets.cuda(rank, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f"[GPU {rank}] Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

    cleanup()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--delta', type=int, default=10)
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size, args.delta), nprocs=world_size, join=True)
