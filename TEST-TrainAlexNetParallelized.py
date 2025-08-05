import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms, models

def setup():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_world_size()

def cleanup():
    dist.destroy_process_group()

def main(delta):
    local_rank, world_size = setup()
    print(f"[GPU {local_rank}] starting training with delta={delta}")

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
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank, shuffle=True)

    # === DataLoader ===
    train_loader = DataLoader(train_dataset, batch_size=128, sampler=train_sampler, num_workers=8, pin_memory=True)

    # === Model ===
    model = models.alexnet().cuda(local_rank)
    model = DDP(model, device_ids=[local_rank])

    # === Loss and Optimizer ===
    criterion = torch.nn.CrossEntropyLoss().cuda(local_rank)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # === Training Loop ===
    for epoch in range(delta):
        train_sampler.set_epoch(epoch)
        model.train()

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.cuda(local_rank, non_blocking=True)
            targets = targets.cuda(local_rank, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 20 == 0 and local_rank == 0:
                print(f"[GPU {local_rank}] Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

    cleanup()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--delta', type=int, default=10)
    args = parser.parse_args()

    main(args.delta)
