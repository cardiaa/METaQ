import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms, models

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'  # oppure un'altra porta libera
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size, delta):
    print(f"[GPU {rank}] starting training with delta={delta}")
    setup(rank, world_size)

    # === ImageNet transforms ===
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # === Dataset ===
    imagenet_path = "/disk1/a.cardia/imagenet"
    train_dataset = datasets.ImageFolder(root=os.path.join(imagenet_path, "train"), transform=transform)

    # === Distributed Sampler ===
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)

    # === DataLoader ===
    train_loader = DataLoader(train_dataset, batch_size=128, sampler=train_sampler, num_workers=8, pin_memory=True)

    # === Model ===
    model = models.alexnet()
    model.cuda(rank)
    model = DDP(model, device_ids=[rank])

    # === Loss and Optimizer ===
    criterion = torch.nn.CrossEntropyLoss().cuda(rank)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # === Training Loop ===
    for epoch in range(delta):
        train_sampler.set_epoch(epoch)
        model.train()

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.cuda(rank, non_blocking=True)
            targets = targets.cuda(rank, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 20 == 0 and rank == 0:
                print(f"[GPU {rank}] Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

    cleanup()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--delta', type=int, default=10)
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size, args.delta), nprocs=world_size, join=True)
