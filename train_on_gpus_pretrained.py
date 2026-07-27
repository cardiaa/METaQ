import argparse
import os
import sys
import math
import itertools
import glob
import tarfile
import json

import torch
import torch.distributed as dist
import torch.nn as nn

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms, models
import webdataset as wds
import socket

from utils.trainer_on_gpus_pretrained import train_and_evaluate
from utils.networks import LeNet5, LeNet5_Original, LeNet300_100


# -------------------------
# DDP utilities
# -------------------------
def ddp_needed(model_name: str) -> bool:
    return model_name in ("AlexNet", "VGG16", "LeNet-5", "LeNet-5 (rotated)", "LeNet300_100")


def setup_ddp():
    dist.init_process_group(backend="nccl")

    # The launcher accepts both torchrun (LOCAL_RANK) and SLURM (SLURM_LOCALID) rank variables.
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
    elif "SLURM_LOCALID" in os.environ:
        local_rank = int(os.environ["SLURM_LOCALID"])
    else:
        raise RuntimeError("Missing LOCAL_RANK/SLURM_LOCALID. Launch with torchrun or proper SLURM env.")

    torch.cuda.set_device(local_rank)
    world_size = dist.get_world_size()
    return local_rank, world_size


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


# -------------------------
# Paths + dataset detection
# -------------------------
def imagenet_paths(data_root: str):
    base = os.path.join(data_root, "imagenet")
    shards_dir = os.path.join(base, "shards")

    shards_train = os.path.join(shards_dir, "train-*.tar")
    shards_val = os.path.join(shards_dir, "val-*.tar")

    folder_train = os.path.join(base, "train")
    folder_val = os.path.join(base, "val")

    has_shards = len(glob.glob(shards_train)) > 0
    has_folders = os.path.isdir(folder_train) and os.path.isdir(folder_val)

    return {
        "base": base,
        "shards_train": shards_train,
        "shards_val": shards_val,
        "folder_train": folder_train,
        "folder_val": folder_val,
        "has_shards": has_shards,
        "has_folders": has_folders,
    }


# -------------------------
# Synset -> idx from shards
# -------------------------
def build_synset_to_idx_from_shards(shards_pattern: str, cache_path: str | None = None):
    """
    Reads the shard tarfiles to extract unique synsets (class names) and builds a synset->index mapping.
    If cache_path is provided, it saves/loads synsets from disk to avoid re-scanning tar files every run.
    """
    if cache_path is not None and os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            synsets = json.load(f)
        synsets = list(synsets)
        synsets = sorted(synsets)
        syn2idx = {s: i for i, s in enumerate(synsets)}
        return syn2idx, synsets

    synsets = set()
    for tar_path in sorted(glob.glob(shards_pattern)):
        with tarfile.open(tar_path) as tf:
            for m in tf.getmembers():
                if m.isfile() and "/" in m.name:
                    synsets.add(m.name.split("/", 1)[0])

    synsets = sorted(synsets)
    if cache_path is not None:
        with open(cache_path, "w") as f:
            json.dump(synsets, f)

    syn2idx = {s: i for i, s in enumerate(synsets)}
    return syn2idx, synsets


# -------------------------
# Model + hyperparameters
# -------------------------
def build_model_and_hparams(model_name: str, device: torch.device, args, local_rank=None):
    # The dictionary stores model-specific defaults so that configuration logging remains uniform across models.
    h = dict(
        criterion=nn.CrossEntropyLoss(),
        criterion_name="CrossEntropy",
        C=None,
        lr=None,
        batch_size=None,
        lambda_reg=0.0,
        alpha=1.0,
        T1_explicit=0.0,
        T2_explicit=0.0,
        subgradient_step=1e5,
        r=2.0,
        w0=0.0,
        BestQuantization_target_acc=99.8,
        final_target_acc=99.7,
        target_zstd_ratio=0.0179,
        min_xi=0.0,
        max_xi=1.0,
        upper_c=None,
        lower_c=1e-2,
        c1=10,
        c2=1000,
        first_best_indices=20,
        accuracy_tollerance=0.2,
        zeta=50000,
        l=0.5,
        n_epochs=1,
        max_iterations=15,
        metrics_interval=1,
        entropy_warmup_epochs=0,
        entropy_every=1,
        check_ddp_sync=False,
        train_optimizer="SGD",
        entropy_optimizer="FISTA",
        pruning="Y",
        QuantizationType="center",
        use_quantization=True,
        sparsity_threshold=1e-3,
        # test_113: perspective reformulation (Frangioni).
        T3_explicit=0.0,          # sparsity weight; L1 push near 0 ~ 2*sqrt(T1*T3)
        mag_prune_ratio=0.5,      # magnitude prune threshold = ratio * min_b|v_b|
        use_perspective=False,
        target_sparsity=0.0,      # if >0: per-layer prune the smallest this fraction of |w|
        sparsity_warmup_epochs=0, # if >0: ramp effective sparsity 0->target over this many epochs
        sparsity_ramp_power=1.0,  # ramp profile: 1.0 linear, <1 concave (gentle near target)
        conv_sparsity=None,       # if set (with fc_sparsity): per-layer target for conv (4D) weights
        fc_sparsity=None,         # if set (with conv_sparsity): per-layer target for FC (2D) weights
        layer_sparsity=None,      # if set: list, one sparsity target per quantized tensor (overrides conv/fc)
    )

    if model_name.startswith("LeNet-5"):
        model = LeNet5_Original().to(device)
        if local_rank is not None:
            model = DDP(model, device_ids=[local_rank])        

        C = 256
        #lambda_reg = 0.0015
        #alpha = 0.533
        r = 1.114
        #bucket_zero = round((C - 1) / 2)
        #w0 = round(r - (bucket_zero + 0.5) * 2 * r * (1 - 1 / C) / (C - 1), 3)
        w0 = -0.11
        h.update(
            C=C,
            lr=0.0007,
            #lambda_reg=lambda_reg,
            #alpha=alpha,
            #T1_explicit=lambda_reg * alpha,
            #T2_explicit=lambda_reg * (1 - alpha),
            # Override
            T1_explicit=0.001,
            T2_explicit=0.0005,            
            r=r,
            w0=w0,
            n_epochs=500,
            train_optimizer="ADAM",
        )
        h["upper_c"] = sum(p.numel() for p in LeNet5().parameters())

    elif model_name == "LeNet300_100":
        model = LeNet300_100().to(device)
        if local_rank is not None:
            model = DDP(model, device_ids=[local_rank])        

        C = 64
        #lambda_reg = 0.0002
        #alpha = 0.6
        r = 2
        bucket_zero = round((C - 1) / 2)
        w0 = round(r - (bucket_zero + 0.5) * 2 * r * (1 - 1 / C) / (C - 1), 3)

        h.update(
            C=C,
            lr=0.001,
            #lambda_reg=lambda_reg,
            #alpha=alpha,
            #T1_explicit=lambda_reg * alpha,
            #T2_explicit=lambda_reg * (1 - alpha),
            # Override
            T1_explicit=0.001,
            T2_explicit=0.0005,             
            r=r,
            w0=w0,
            n_epochs=100,
            train_optimizer="ADAM",
        )
        h["upper_c"] = sum(p.numel() for p in LeNet300_100().parameters())

    elif model_name == "AlexNet":
        if local_rank is None:
            raise RuntimeError("AlexNet requires DDP setup (local_rank is None).")

        # The pretrained branch loads the standard torchvision ImageNet-1K AlexNet
        # checkpoint.  The randomly initialized branch preserves the previous
        # experimental setup used for training AlexNet from scratch.
        if args.pretrained == "Y":
            model = models.alexnet(weights=None)
            checkpoint = torch.load(args.pretrained_checkpoint, map_location="cpu")
            model.load_state_dict(checkpoint)
        else:
            model = models.alexnet(weights=None)
            model.classifier[6] = nn.Linear(4096, 1000)

        model = model.to(device, memory_format=torch.channels_last)
        model = DDP(model, device_ids=[local_rank])

        h.update(
            C=16,
            # Pretrained fine-tuning uses a small learning rate to avoid
            # destroying the already trained ImageNet representation.
            lr=1e-4 if args.pretrained == "Y" else 1e-1,
            #batch_size=2048,
            batch_size=128,  # Leonardo default; command-line arguments can override it.
            #lambda_reg=5e-4,
            #alpha=0.99999,
            T1_explicit=1e-3,
            T2_explicit=1e-6,
            r=1.51,
            w0=0.013,
            n_epochs=20,
            train_optimizer="SGD",
        )
        h["upper_c"] = sum(p.numel() for p in model.parameters())

    elif model_name == "VGG16":
        if local_rank is None:
            raise RuntimeError("VGG16 requires DDP setup (local_rank is None).")

        model = models.vgg16(weights=None)
        model.classifier[6] = nn.Linear(4096, 1000)
        model = model.to(device)
        model = DDP(model, device_ids=[local_rank])

        C = 8
        #lambda_reg = 0.0005
        #alpha = 0.9
        r = 2
        bucket_zero = round((C - 1) / 2)
        w0 = round(r - (bucket_zero + 0.5) * 2 * r * (1 - 1 / C) / (C - 1), 3)

        h.update(
            C=C,
            lr=0.01,
            batch_size=512,
            #lambda_reg=lambda_reg,
            #alpha=alpha,
            #T1_explicit=lambda_reg * alpha,
            #T2_explicit=lambda_reg * (1 - alpha),
            # Override
            T1_explicit=0.001,
            T2_explicit=0.0005,             
            r=r,
            w0=w0,
            n_epochs=20,
            train_optimizer="SGD",
        )
        h["upper_c"] = sum(p.numel() for p in model.parameters())

    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    return model, h


# -------------------------
# Data loading: MNIST
# -------------------------
def load_mnist_lenet5(model_name: str, data_root: str):
    if model_name.endswith("(rotated)"):
        transform_train = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomRotation(10),
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
        ])

    trainset = datasets.MNIST(root=data_root, train=True, download=True, transform=transform_train)
    testset = datasets.MNIST(root=data_root, train=False, download=True, transform=transform_test)
    return trainset, testset


def load_mnist_lenet300(data_root: str):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    trainset = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    testset = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)
    return trainset, testset

"""
# -------------------------
# DDP node splitter for WebDataset (no splitting, all processes can read all shards)
# -------------------------
def identity_nodesplitter(urls):
    return urls
"""

# -------------------------
# DDP node splitter for WebDataset (split shards across processes, but no shard-level shuffling)
# -------------------------
def split_by_rank(urls):
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    # The splitter returns only the URLs assigned to the current distributed rank, using round-robin assignment.
    return itertools.islice(urls, rank, None, world)

"""
# -------------------------
# Split shards among DataLoader workers inside the same rank.
# This avoids duplicated work when num_workers > 1.
# -------------------------
def split_by_worker(urls):
    info = torch.utils.data.get_worker_info()
    if info is None:
        return urls

    worker_id = info.id
    num_workers = info.num_workers
    return itertools.islice(urls, worker_id, None, num_workers)
"""
"""
# -------------------------
# First split across distributed ranks, 
# then split across workers inside each rank.
# -------------------------
def split_by_rank_and_worker(urls):
    return split_by_worker(split_by_rank(urls))
"""

# -------------------------
# Data loading: ImageNet (shards or folders)
# -------------------------
def load_imagenet_dataloaders(batch_size, data_root, local_rank, world_size, train_workers, val_workers):
    t_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    t_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    p = imagenet_paths(data_root)

    # --- Leonardo: WebDataset shards ---
    if p["has_shards"]:
        # Creates synset->idx mapping from shard contents. 
        # Important: this ensures that all processes have the same mapping, 
        # even if the order of shards is not guaranteed to be the same across processes.
        cache_path = os.path.join(p["base"], "shards", "synsets.json")

        if dist.is_initialized():
            rank = dist.get_rank()
            obj = [None]
            if rank == 0:
                _, synsets = build_synset_to_idx_from_shards(p["shards_train"], cache_path=cache_path)
                obj[0] = synsets
            dist.broadcast_object_list(obj, src=0)
            synsets = obj[0]
            syn2idx = {s: i for i, s in enumerate(synsets)}
        else:
            syn2idx, _ = build_synset_to_idx_from_shards(p["shards_train"], cache_path=cache_path)

        def key_to_label(key: str):
            syn = key.split("/", 1)[0]

            # Standard case: synset type n01440764
            if syn in syn2idx:
                return torch.tensor(syn2idx[syn], dtype=torch.long)

            # Leonardo shards case: numeric key type "490" / "922"
            if syn.isdigit():
                k = int(syn)
                # The mapper treats values in [0, 999] as zero-based labels and values in [1, 1000] as one-based labels.
                if 0 <= k <= 999:
                    return torch.tensor(k, dtype=torch.long)
                if 1 <= k <= 1000:
                    return torch.tensor(k - 1, dtype=torch.long)

            raise KeyError(f"Unrecognized sample key format: {key}")

        # ImageNet sizes (standard)
        train_size = 1281167
        val_size = 50000
        steps_per_epoch = max(1, train_size // (batch_size * max(1, world_size)))

        train_urls = sorted(glob.glob(p["shards_train"]))
        val_urls   = sorted(glob.glob(p["shards_val"]))
        
        # ---------------------------------------------------------------------
        # DEBUG
        # Diagnostic print after building the dataset but before training, 
        # to verify that the expected number of shards and samples are detected, 
        # and that the batch size and world size are correctly accounted 
        # for in the steps per epoch.
        if (not dist.is_initialized()) or dist.get_rank() == 0:
            print(
                f"[ImageNet loader] batch_size={batch_size}, world_size={world_size}, "
                f"steps_per_epoch={steps_per_epoch}, "
                f"train_shards={len(train_urls)}, val_shards={len(val_urls)}, "
                f"train_workers={train_workers}, val_workers={val_workers}",
                flush=True
            ) 
        # ---------------------------------------------------------------------     

        # Note: in the shards the key is of the form "n01440764/xxx.JPEG", 
        # so the synset is the first part before "/".
        train_ds = (
            wds.WebDataset(
                train_urls,
                shardshuffle=1000,
                nodesplitter=split_by_rank,
                workersplitter=wds.split_by_worker,
            )
            .shuffle(10000)
            .decode("pil")
            .to_tuple("__key__", "jpg;JPEG;jpeg;png")
            .map_tuple(lambda k: k, t_train)
            .map(lambda k_img: (k_img[1], key_to_label(k_img[0])))
            .repeat()
            .batched(batch_size, partial=False)
        )

        val_ds = (
            wds.WebDataset(
                val_urls,
                shardshuffle=False,
                nodesplitter=split_by_rank,
                workersplitter=wds.split_by_worker,
                empty_check=False,
            )
            .decode("pil")
            .to_tuple("__key__", "jpg;JPEG;jpeg;png")
            .map_tuple(lambda k: k, t_val)
            .map(lambda k_img: (k_img[1], key_to_label(k_img[0])))
            .batched(batch_size, partial=True)
        )

        trainloader = wds.WebLoader(train_ds, batch_size=None, num_workers=train_workers, pin_memory=True)
        testloader = wds.WebLoader(val_ds, batch_size=None, num_workers=val_workers, pin_memory=True)
        train_sampler = None

        return trainloader, testloader, train_sampler, steps_per_epoch

    # --- 4-GPU machine: ImageFolder ---
    if p["has_folders"]:
        train_dataset = datasets.ImageFolder(p["folder_train"], transform=t_train)
        val_dataset = datasets.ImageFolder(p["folder_val"], transform=t_val)
        steps_per_epoch = max(1, len(train_dataset) // (batch_size * max(1, world_size)))

        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=local_rank,
            shuffle=True,
            drop_last=True,
        )

        trainloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=train_workers,
            pin_memory=True,
        )

        val_sampler = None
        if dist.is_initialized() and world_size > 1:
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=world_size,
                rank=local_rank,
                shuffle=False,
                drop_last=False,
            )

        testloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            shuffle=False,
            num_workers=val_workers,
            pin_memory=True,
        )

        return trainloader, testloader, train_sampler, steps_per_epoch

    raise RuntimeError(
        f"ImageNet not found. Expected either shards in {p['shards_train']} or folders in {p['folder_train']}."
    )


# -------------------------
# Logging
# -------------------------
def print_config(model_name, args, h, local_rank_to_print):
    if local_rank_to_print != 0:
        return

    print("=================================================================", flush=True)
    print("==================== PARAMETER CONFIGURATION ====================", flush=True)
    print("=================================================================", flush=True)
    print(f"model={model_name}", flush=True)
    print(f"criterion={h['criterion_name']}", flush=True)
    print(f"C={h['C']}", flush=True)
    print(f"layer_C={h['layer_C']}", flush=True)
    print(f"delta={args.delta}", flush=True)
    print(f"gamma={args.gamma}", flush=True)    
    print(f"lr={h['lr']}", flush=True)
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if h["batch_size"] is not None:
        print(f"batch_size_per_gpu={h['batch_size']}", flush=True)
        print(f"global_batch_size={h['batch_size'] * world_size}", flush=True)
    print(f"T1={h['T1_explicit']}", flush=True)
    print(f"T2={h['T2_explicit']}", flush=True)
    print(f"T3={h['T3_explicit']}", flush=True)
    print(f"use_perspective={h['use_perspective']}", flush=True)
    print(f"mag_prune_ratio={h['mag_prune_ratio']}", flush=True)
    print(f"target_sparsity={h['target_sparsity']}", flush=True)
    print(f"sparsity_warmup_epochs={h['sparsity_warmup_epochs']}", flush=True)
    print(f"sparsity_ramp_power={h['sparsity_ramp_power']}", flush=True)
    print(f"conv_sparsity={h['conv_sparsity']}", flush=True)
    print(f"fc_sparsity={h['fc_sparsity']}", flush=True)
    print(f"layer_sparsity={h['layer_sparsity']}", flush=True)
    print(f"sparsity_schedule={h['sparsity_schedule']}", flush=True)
    print(f"freeze_mask={h['freeze_mask']}", flush=True)
    print(f"train_sparse={h['train_sparse']}", flush=True)
    print(f"train_centroids={h['train_centroids']}", flush=True)
    print(f"centroid_lr_scale={h['centroid_lr_scale']}", flush=True)
    print(f"centroid_kmeans_iterations={h['centroid_kmeans_iterations']}", flush=True)
    print(f"centroid_freeze_epoch={h['centroid_freeze_epoch']}", flush=True)
    print(f"adiabatic_accuracy_target={h['adiabatic_accuracy_target']}", flush=True)
    print(f"adiabatic_accuracy_tolerance={h['adiabatic_accuracy_tolerance']}", flush=True)
    print(f"adiabatic_step={h['adiabatic_step']}", flush=True)
    print(f"adiabatic_backoff={h['adiabatic_backoff']}", flush=True)
    print(f"adiabatic_patience={h['adiabatic_patience']}", flush=True)
    print(f"metrics_interval={h['metrics_interval']}", flush=True)
    print(f"entropy_warmup_epochs={h['entropy_warmup_epochs']}", flush=True)
    print(f"entropy_every={h['entropy_every']}", flush=True)
    print(f"check_ddp_sync={h['check_ddp_sync']}", flush=True)
    print(f"subgradient_step={h['subgradient_step']}", flush=True)
    print(f"w0={h['w0']}", flush=True)
    print(f"r={h['r']}", flush=True)
    print(f"BestQuantization_target_acc={h['BestQuantization_target_acc']}", flush=True)
    print(f"final_target_acc={h['final_target_acc']}", flush=True)
    print(f"target_zstd_ratio={h['target_zstd_ratio']}", flush=True)
    print(f"min_xi={h['min_xi']}", flush=True)
    print(f"max_xi={h['max_xi']}", flush=True)
    print(f"upper_c={h['upper_c']}", flush=True)
    print(f"lower_c={h['lower_c']}", flush=True)
    print(f"c1={h['c1']}", flush=True)
    print(f"c2={h['c2']}", flush=True)
    print(f"first_best_indices={h['first_best_indices']}", flush=True)
    print(f"accuracy_tollerance={h['accuracy_tollerance']}", flush=True)
    print(f"zeta={h['zeta']}", flush=True)
    print(f"l={h['l']}", flush=True)
    print(f"n_epochs={h['n_epochs']}", flush=True)
    print(f"max_iterations={h['max_iterations']}", flush=True)
    print(f"train_optimizer={h['train_optimizer']}", flush=True)
    print(f"entropy_optimizer={h['entropy_optimizer']}", flush=True)
    print(f"pruning={h['pruning']}", flush=True)
    print(f"QuantizationType={h['QuantizationType']}", flush=True)
    print(f"use_quantization={h['use_quantization']}", flush=True)
    print(f"sparsity_threshold={h['sparsity_threshold']}", flush=True)
    print("-" * 60, flush=True)
    print("", flush=True)


def print_allocated_gpus_once(device: torch.device):
    """
    Each process gathers the hostname and GPU info of all processes, 
    and rank 0 prints a summary grouped by hostname.
     - This is useful to verify that the expected GPUs are allocated and visible to each process, 
     especially in multi-node setups where GPU visibility can be tricky.
     - The function is designed to be called once at the beginning of training, 
     to avoid cluttering logs with repeated GPU info every epoch.
    """
    rank = dist.get_rank()
    hostname = socket.gethostname()

    if device.type == "cuda":
        local_cuda = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(local_cuda)
        msg = f"cuda:{local_cuda} ({gpu_name})"
    else:
        msg = str(device)

    payload = (hostname, msg)

    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, payload)

    if rank == 0:
        by_node = {}
        for host, gpu in gathered:
            by_node.setdefault(host, []).append(gpu)

        # Rank 0 prints a compact, grouped GPU allocation summary.
        for host in sorted(by_node):
            print(f"### {host}", flush=True)
            for gpu in sorted(by_node[host]):
                print(gpu, flush=True)
            print("", flush=True)    


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--delta", type=float, required=True, help="Value of delta")
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Multiplier for the delta-controlled pruning deadzone",
    )    
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=["LeNet-5", "LeNet-5 (rotated)", "LeNet300_100", "AlexNet", "VGG16"],
        help="Name of the model to train",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="./data",
        help="Root directory for datasets. Expects data_root/imagenet/(shards|train,val)",
    )
    parser.add_argument("--train_workers", type=int, default=1, help="Number of DataLoader workers for training")
    parser.add_argument("--val_workers", type=int, default=2, help="Number of DataLoader workers for validation")
    parser.add_argument("--n_epochs", type=int, default=None, help="Override number of epochs (if set)")
    parser.add_argument("--batch_size", type=int, default=None, help="Override per-GPU batch size (if set)")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate (if set)")
    parser.add_argument("--T1", type=float, default=None, help="Override L2 weight decay term; 0 disables it")
    parser.add_argument("--T2", type=float, default=None, help="Override entropy term; 0 disables it")
    parser.add_argument("--T3", type=float, default=None, help="Perspective sparsity weight; L1 push near 0 ~ 2*sqrt(T1*T3)")
    parser.add_argument("--mag_prune_ratio", type=float, default=None, help="Magnitude prune threshold = ratio * min_b|v_b|")
    parser.add_argument("--perspective", type=str, default="N", choices=["Y", "N"], help="Enable the perspective reformulation (test_113)")
    parser.add_argument("--flat_schedule", type=str, default="N", choices=["Y", "N"], help="Hold BOTH lr and T2 constant for the whole run: no cosine tail, no T2 ramp (test_133)")
    parser.add_argument("--dual_step", type=float, default=None, help="Ascent step for the entropy dual on the layer-size-normalized supergradient (test_134)")
    parser.add_argument("--prox", type=str, default="N", choices=["Y", "N"], help="Apply phi as a proximal operator on the weights instead of summing beta* into the loss gradient (test_135)")
    parser.add_argument("--prox_gamma", type=float, default=None, help="Step of the proximal operator; sets the entropy displacement independently of the learning rate (test_135)")
    parser.add_argument("--prox_start_epoch", type=int, default=0, help="First epoch (0-based) at which the proximal step runs; decoupled from entropy_warmup_epochs so the sparsity ramp is unaffected (test_139)")
    parser.add_argument("--sparsity_schedule", type=str, default=None, help="Iterative prune-and-heal schedule, replacing the smooth ramp. Stages ';'-separated, each 'reach:hold:s1,...,s8' with 1-based epochs (test_143)")
    parser.add_argument("--freeze_mask", type=str, default="N", choices=["Y", "N"], help="Freeze the pruned index set per plateau (Deep-Compression style) instead of recomputing |w|<=thr every epoch (test_144)")
    parser.add_argument("--train_sparse", type=str, default="N", choices=["Y", "N"], help="Optimize the sparse subnetwork directly: hold pruned weights at zero with zero gradient, update only survivors (test_146)")
    parser.add_argument("--quantization", type=str, default="Y", choices=["Y", "N"], help="Enable quantization in training, evaluation, and compression metrics. Set N for an FP32 pruning-only control (test_155)")
    parser.add_argument("--layer_C", type=str, default=None, help="Comma-separated quantization levels per weight tensor. Deep-Compression control for AlexNet: 256,256,256,256,256,32,32,32")
    parser.add_argument("--train_centroids", type=str, default="N", choices=["Y", "N"], help="Freeze cluster assignments and train shared per-layer centroids by summing gradients within each bucket (test_150)")
    parser.add_argument("--centroid_lr_scale", type=float, default=1.0, help="Learning-rate multiplier applied to summed centroid gradients (test_151)")
    parser.add_argument("--centroid_kmeans_iterations", type=int, default=0, help="Scalar Lloyd iterations before freezing centroid assignments; 0 keeps the linear grid (test_153)")
    parser.add_argument("--centroid_freeze_epoch", type=int, default=0, help="1-based epoch that converts dynamic QAT into a fixed codebook; 0 freezes at grid initialization")
    parser.add_argument("--adiabatic_accuracy_target", type=float, default=None, help="Enable accuracy-controlled sparsity and advance only at/above this sparse accuracy")
    parser.add_argument("--adiabatic_accuracy_tolerance", type=float, default=0.2, help="Hysteresis below the adiabatic accuracy target before sparsity is rolled back")
    parser.add_argument("--adiabatic_step", type=float, default=0.02, help="Fraction of the final per-layer sparsity vector added after each accepted plateau")
    parser.add_argument("--adiabatic_backoff", type=float, default=0.04, help="Fraction of final sparsity removed when accuracy falls below the hysteresis band")
    parser.add_argument("--adiabatic_patience", type=int, default=2, help="Consecutive evaluations at target accuracy required before increasing sparsity")
    parser.add_argument("--target_sparsity", type=float, default=None, help="If >0: per-layer prune the smallest this fraction of |w| (overrides mag_prune_ratio)")
    parser.add_argument("--sparsity_warmup_epochs", type=int, default=None, help="If >0: ramp effective sparsity 0->target linearly over this many epochs")
    parser.add_argument("--sparsity_ramp_power", type=float, default=None, help="Ramp profile exponent: 1.0 linear, <1 concave (gentle increments near target)")
    parser.add_argument("--conv_sparsity", type=float, default=None, help="Per-layer target sparsity for conv (4D) weights; use with --fc_sparsity")
    parser.add_argument("--fc_sparsity", type=float, default=None, help="Per-layer target sparsity for FC (2D) weights; use with --conv_sparsity")
    parser.add_argument("--layer_sparsity", type=str, default=None, help="Comma-separated per-layer sparsity targets, one per quantized tensor in order (overrides conv/fc). E.g. 0.16,0.62,0.65,0.63,0.63,0.91,0.91,0.75")
    parser.add_argument("--max_iterations", type=int, default=None, help="Override FISTA/PBM iterations (if set)")
    parser.add_argument("--metrics_interval", type=int, default=1, help="Evaluate/compress every N epochs")
    parser.add_argument("--entropy_warmup_epochs", type=int, default=0, help="Epochs with entropy term disabled")
    parser.add_argument("--entropy_every", type=int, default=1, help="Apply entropy term every N optimizer steps")
    parser.add_argument("--check_ddp_sync", action="store_true", help="Print a parameter checksum range after evaluation")
    parser.add_argument(
        "--epoch_fraction",
        type=float,
        default=1.0,
        help="Fraction of ImageNet seen per epoch, e.g. 0.25 means 25%"
    )
    parser.add_argument(
        "--C",
        type=int,
        default=None,
        help="Override number of quantization buckets"
    )    
    parser.add_argument(
        "--pretrained",
        type=str,
        default="N",
        choices=["Y", "N"],
        help="Use torchvision ImageNet-1K pretrained weights for AlexNet when set to Y."
    )
    parser.add_argument(
        "--pretrained_checkpoint",
        type=str,
        default="/leonardo_work/IscrC_ObCTDoNN/acardia0/alexnet_checkpoints/alexnet-owt-7be5be79.pth",
        help="Path to a local AlexNet pretrained checkpoint"
    )    
    args = parser.parse_args()
    if args.gamma < 0:
        raise ValueError("--gamma must be >= 0.")    

    # CPU thread control
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"

    model_name = args.model_name

    local_rank = None
    world_size = 1

    if ddp_needed(model_name):
        local_rank, world_size = setup_ddp()
        device = torch.device(f"cuda:{local_rank}")
        #print(f"[GPU {local_rank}] Using device {device} ({torch.cuda.get_device_name(device)})", flush=True)
        print_allocated_gpus_once(device)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print(f"Using device {device} ({torch.cuda.get_device_name(device)})", flush=True)
        else:
            print("Using CPU.", flush=True)

    model, h = build_model_and_hparams(model_name, device, args, local_rank=local_rank)

    if args.n_epochs is not None:
        h["n_epochs"] = args.n_epochs
    if args.batch_size is not None:
        h["batch_size"] = args.batch_size
    if args.lr is not None:
        h["lr"] = args.lr
    if args.T1 is not None:
        h["T1_explicit"] = args.T1
    if args.T2 is not None:
        h["T2_explicit"] = args.T2
    if args.T3 is not None:
        h["T3_explicit"] = args.T3
    if args.mag_prune_ratio is not None:
        h["mag_prune_ratio"] = args.mag_prune_ratio
    if args.target_sparsity is not None:
        h["target_sparsity"] = args.target_sparsity
    if args.sparsity_warmup_epochs is not None:
        h["sparsity_warmup_epochs"] = args.sparsity_warmup_epochs
    if args.sparsity_ramp_power is not None:
        h["sparsity_ramp_power"] = args.sparsity_ramp_power
    if args.conv_sparsity is not None:
        h["conv_sparsity"] = args.conv_sparsity
    if args.fc_sparsity is not None:
        h["fc_sparsity"] = args.fc_sparsity
    if args.layer_sparsity is not None:
        h["layer_sparsity"] = [float(s) for s in args.layer_sparsity.split(",") if s.strip() != ""]
    h["use_perspective"] = (args.perspective == "Y")
    h["flat_schedule"] = (args.flat_schedule == "Y")
    h["dual_step"] = 0.5 if args.dual_step is None else args.dual_step
    h["use_prox"] = (args.prox == "Y")
    h["prox_gamma"] = 1e-7 if args.prox_gamma is None else args.prox_gamma
    h["prox_start_epoch"] = args.prox_start_epoch
    h["sparsity_schedule"] = args.sparsity_schedule if args.sparsity_schedule else None
    h["freeze_mask"] = (args.freeze_mask == "Y")
    h["train_sparse"] = (args.train_sparse == "Y")
    h["use_quantization"] = (args.quantization == "Y")
    h["train_centroids"] = (args.train_centroids == "Y")
    h["centroid_lr_scale"] = args.centroid_lr_scale
    h["centroid_kmeans_iterations"] = args.centroid_kmeans_iterations
    h["centroid_freeze_epoch"] = args.centroid_freeze_epoch
    h["layer_C"] = (
        [int(s) for s in args.layer_C.split(",") if s.strip() != ""]
        if args.layer_C is not None
        else None
    )
    h["adiabatic_accuracy_target"] = args.adiabatic_accuracy_target
    h["adiabatic_accuracy_tolerance"] = args.adiabatic_accuracy_tolerance
    h["adiabatic_step"] = args.adiabatic_step
    h["adiabatic_backoff"] = args.adiabatic_backoff
    h["adiabatic_patience"] = args.adiabatic_patience
    if args.max_iterations is not None:
        h["max_iterations"] = args.max_iterations
    if args.C is not None:
        h["C"] = args.C        
    if args.metrics_interval < 1:
        raise ValueError("--metrics_interval must be >= 1.")
    if args.entropy_warmup_epochs < 0:
        raise ValueError("--entropy_warmup_epochs must be >= 0.")
    if args.entropy_every < 1:
        raise ValueError("--entropy_every must be >= 1.")
    h["metrics_interval"] = args.metrics_interval
    h["entropy_warmup_epochs"] = args.entropy_warmup_epochs
    h["entropy_every"] = args.entropy_every
    h["check_ddp_sync"] = args.check_ddp_sync

    # Data
    if model_name.startswith("LeNet-5"):
        trainset, testset = load_mnist_lenet5(model_name, args.data_root)

        if ddp_needed(model_name):
            train_sampler = DistributedSampler(
                trainset,
                num_replicas=world_size,
                rank=local_rank,
                shuffle=True,
            )

            test_sampler = DistributedSampler(
                testset,
                num_replicas=world_size,
                rank=local_rank,
                shuffle=False,
            )

            trainloader = DataLoader(
                trainset,
                batch_size=64,
                sampler=train_sampler,
                num_workers=0,
                drop_last=True,
            )

            testloader = DataLoader(
                testset,
                batch_size=1000,
                sampler=test_sampler,
                num_workers=0,
            )
        else:
            trainloader = DataLoader(trainset, batch_size=64, shuffle=True, drop_last=True, num_workers=0)
            testloader = DataLoader(testset, batch_size=1000, shuffle=False, num_workers=0)
            train_sampler = None

        steps_per_epoch = None

    elif model_name == "LeNet300_100":
        trainset, testset = load_mnist_lenet300(args.data_root)
        trainloader = DataLoader(trainset, batch_size=64, shuffle=True, drop_last=True, num_workers=0)
        testloader = DataLoader(testset, batch_size=1000, shuffle=False, num_workers=0)
        train_sampler = None
        steps_per_epoch = None

    else:
        trainloader, testloader, train_sampler, steps_per_epoch = load_imagenet_dataloaders(
            batch_size=h["batch_size"],
            data_root=args.data_root,
            local_rank=local_rank,
            world_size=world_size,
            train_workers=args.train_workers,
            val_workers=args.val_workers,
        )
        if steps_per_epoch is not None:
            if not (0 < args.epoch_fraction <= 1):
                raise ValueError("--epoch_fraction must be in (0, 1].")

            steps_per_epoch = max(1, int(steps_per_epoch * args.epoch_fraction))

            if dist.is_initialized() and dist.get_rank() == 0:
                print(
                    f"[ImageNet loader] using epoch_fraction={args.epoch_fraction}, "
                    f"effective_steps_per_epoch={steps_per_epoch}",
                    flush=True
                )        

    if not ddp_needed(model_name):
        print_config(model_name, args, h, 0)
    else:
        if dist.is_initialized() and dist.get_rank() == 0:
            print_config(model_name, args, h, 0)

    # Training
    train_and_evaluate(
        model=model,
        model_name=model_name,
        criterion=h["criterion"],
        C=h["C"],
        lr=h["lr"],
        lambda_reg=h["lambda_reg"],
        alpha=h["alpha"],
        T1_explicit=h["T1_explicit"],
        T2_explicit=h["T2_explicit"],
        subgradient_step=h["subgradient_step"],
        w0=h["w0"],
        r=h["r"],
        first_best_indices=h["first_best_indices"],
        BestQuantization_target_acc=h["BestQuantization_target_acc"],
        final_target_acc=h["final_target_acc"],
        target_zstd_ratio=h["target_zstd_ratio"],
        min_xi=h["min_xi"],
        max_xi=h["max_xi"],
        upper_c=h["upper_c"],
        lower_c=h["lower_c"],
        c1=h["c1"],
        c2=h["c2"],
        zeta=h["zeta"],
        l=h["l"],
        n_epochs=h["n_epochs"],
        max_iterations=h["max_iterations"],
        device=device,
        train_optimizer=h["train_optimizer"],
        entropy_optimizer=h["entropy_optimizer"],
        trainloader=trainloader,
        testloader=testloader,
        train_sampler=train_sampler,
        steps_per_epoch=steps_per_epoch,
        delta=args.delta,
        pruning=h["pruning"],
        QuantizationType=h["QuantizationType"],
        sparsity_threshold=h["sparsity_threshold"],
        accuracy_tollerance=h["accuracy_tollerance"],
        gamma=args.gamma,
        metrics_interval=h["metrics_interval"],
        entropy_warmup_epochs=h["entropy_warmup_epochs"],
        entropy_every=h["entropy_every"],
        check_ddp_sync=h["check_ddp_sync"],
        T3_explicit=h["T3_explicit"],
        mag_prune_ratio=h["mag_prune_ratio"],
        use_perspective=h["use_perspective"],
        target_sparsity=h["target_sparsity"],
        sparsity_warmup_epochs=h["sparsity_warmup_epochs"],
        sparsity_ramp_power=h["sparsity_ramp_power"],
        conv_sparsity=h["conv_sparsity"],
        fc_sparsity=h["fc_sparsity"],
        layer_sparsity=h["layer_sparsity"],
        flat_schedule=h["flat_schedule"],
        dual_step=h["dual_step"],
        use_prox=h["use_prox"],
        prox_gamma=h["prox_gamma"],
        prox_start_epoch=h["prox_start_epoch"],
        sparsity_schedule=h["sparsity_schedule"],
        freeze_mask=h["freeze_mask"],
        train_sparse=h["train_sparse"],
        use_quantization=h["use_quantization"],
        layer_C=h["layer_C"],
        train_centroids=h["train_centroids"],
        centroid_lr_scale=h["centroid_lr_scale"],
        centroid_kmeans_iterations=h["centroid_kmeans_iterations"],
        centroid_freeze_epoch=h["centroid_freeze_epoch"],
        adiabatic_accuracy_target=h["adiabatic_accuracy_target"],
        adiabatic_accuracy_tolerance=h["adiabatic_accuracy_tolerance"],
        adiabatic_step=h["adiabatic_step"],
        adiabatic_backoff=h["adiabatic_backoff"],
        adiabatic_patience=h["adiabatic_patience"],
    )

    if ddp_needed(model_name):
        cleanup_ddp()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        rank = int(os.environ.get("RANK", "0"))
        if rank == 0:
            raise
        else:
            print(f"[rank {rank}] failed; traceback suppressed: {type(e).__name__}: {e}", flush=True)
            sys.exit(1)
