"""
Per-layer pruning sensitivity of the pretrained AlexNet (no retraining).

Motivation: every experiment so far has used the per-layer pruning rates COPIED
from Deep Compression (0.16, 0.62, 0.65, 0.63, 0.63, 0.91, 0.91, 0.75).  Those
rates were tuned by Han et al. for their own setup; nobody has ever checked
whether they are the right allocation for the network and the deploy pipeline we
actually use.  If our sensitivity profile differs, we are protecting the wrong
layers, and every run inherits that mistake.

Method (the standard one, cf. Han et al. 2015): take the pretrained dense model,
prune ONE layer at a time at increasing rates, and measure top-1 accuracy on the
validation set WITHOUT any retraining.  The knee of each curve says how much that
layer tolerates.  No training is involved, so this is minutes of forward passes
rather than a campaign of runs.

Single GPU, single process: no DDP, no sharding by rank.

    python sensitivity_analysis.py --data_root <dir> --pretrained_checkpoint <pth>

Output: one line per (layer, rate) plus a summary table, and optionally a CSV.
"""
import argparse
import glob
import json
import os
import time

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms


def build_val_loader(data_root, batch_size, workers):
    """Validation loader only.  Mirrors the transforms used in training.

    The paths and the synset->index mapping are taken from the trainer itself
    (`imagenet_paths`, `build_synset_to_idx_from_shards`) rather than rebuilt
    here: the layout is data_root/imagenet/shards/val-*.tar, and duplicating
    that knowledge is exactly how the first version of this script ended up
    globbing the wrong directory and silently falling through to ImageFolder.
    """
    t_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    from train_on_gpus_pretrained import (
        imagenet_paths, build_synset_to_idx_from_shards,
    )

    p = imagenet_paths(data_root)
    val_urls = sorted(glob.glob(p["shards_val"]))

    if val_urls:
        import webdataset as wds

        cache_path = os.path.join(p["base"], "shards", "synsets.json")
        syn2idx, _ = build_synset_to_idx_from_shards(
            p["shards_train"], cache_path=cache_path,
        )

        def key_to_label(key):
            head = key.split("/")[0]
            # Leonardo shards use numeric keys ("490"); synset keys ("n01440764")
            # go through the mapping.
            if head.isdigit():
                return int(head)
            return syn2idx[head]

        val_ds = (
            wds.WebDataset(val_urls, shardshuffle=False, empty_check=False)
            .decode("pil")
            .to_tuple("__key__", "jpg;JPEG;jpeg;png")
            .map_tuple(lambda k: k, t_val)
            .map(lambda ki: (ki[1], key_to_label(ki[0])))
            .batched(batch_size, partial=True)
        )
        print(f"[loader] shards val: {len(val_urls)} file da {p['shards_val']}",
              flush=True)
        return wds.WebLoader(val_ds, batch_size=None, num_workers=workers,
                             pin_memory=True)

    if p["has_folders"]:
        from torch.utils.data import DataLoader
        from torchvision import datasets
        print(f"[loader] ImageFolder da {p['folder_val']}", flush=True)
        ds = datasets.ImageFolder(p["folder_val"], transform=t_val)
        return DataLoader(ds, batch_size=batch_size, shuffle=False,
                          num_workers=workers, pin_memory=True)

    raise FileNotFoundError(
        f"Nessun dato di validazione trovato: ne shard in {p['shards_val']}, "
        f"ne cartelle in {p['folder_val']}. Controlla --data_root "
        f"(atteso: la directory che CONTIENE imagenet/)."
    )


@torch.inference_mode()
def top1(model, loader, device, max_batches=None):
    model.eval()
    correct = torch.zeros((), device=device, dtype=torch.long)
    total = torch.zeros((), device=device, dtype=torch.long)
    ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                         enabled=(device.type == "cuda"))
    with ctx:
        for i, (x, y) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if isinstance(y, list):
                y = torch.tensor(y, device=device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum()
            total += y.numel()
    return 100.0 * correct.item() / max(1, total.item())


def prune_layer_(weight, rate):
    """Zero the `rate` fraction of smallest-|w| entries.  Returns a backup."""
    backup = weight.detach().clone()
    if rate <= 0.0:
        return backup
    flat = weight.detach().reshape(-1)
    k = int(rate * flat.numel())
    k = max(1, min(flat.numel() - 1, k))
    thr = torch.kthvalue(flat.abs().float(), k).values
    weight.detach().masked_fill_(flat.abs().view_as(weight) <= thr, 0.0)
    return backup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str,
                    default="/leonardo_work/IscrC_ObCTDoNN/acardia0/datasets")
    ap.add_argument("--pretrained_checkpoint", type=str,
                    default="/leonardo_work/IscrC_ObCTDoNN/acardia0/"
                            "alexnet_checkpoints/alexnet-owt-7be5be79.pth")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max_batches", type=int, default=None,
                    help="Limit validation batches (quick smoke run).")
    ap.add_argument("--rates", type=str,
                    default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95",
                    help="Comma-separated pruning rates to sweep per layer.")
    ap.add_argument("--csv", type=str, default="sensitivity.csv")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rates = [float(r) for r in args.rates.split(",") if r.strip()]

    model = models.alexnet(weights=None)
    model.load_state_dict(torch.load(args.pretrained_checkpoint,
                                     map_location="cpu"))
    model = model.to(device, memory_format=torch.channels_last)

    loader = build_val_loader(args.data_root, args.batch_size, args.workers)

    # Same tensors, in the same order, that the trainer quantizes and prunes.
    layers = [(n, p) for n, p in model.named_parameters() if p.dim() in (2, 4)]

    print("=" * 78, flush=True)
    t0 = time.time()
    base = top1(model, loader, device, args.max_batches)
    print(f"baseline (dense, no pruning): {base:.2f}%   "
          f"[{time.time() - t0:.0f}s per evaluation]", flush=True)
    print("=" * 78, flush=True)

    # Deep Compression's rates, in the trainer's tensor order, for reference.
    dc_rates = [0.16, 0.62, 0.65, 0.63, 0.63, 0.91, 0.91, 0.75]

    rows = []
    for li, (name, p) in enumerate(layers):
        n = p.numel()
        print(f"\n--- [{li}] {name}  ({n:,} pesi, {100*n/61090496:.2f}% del "
              f"totale)   tasso DC = {dc_rates[li]:.2f}", flush=True)
        for r in rates:
            backup = prune_layer_(p, r)
            acc = top1(model, loader, device, args.max_batches)
            p.detach().copy_(backup)          # restore before the next rate
            drop = base - acc
            rows.append((li, name, n, r, acc, drop))
            flag = "  <-- tasso DC" if abs(r - dc_rates[li]) < 0.051 else ""
            print(f"    rate {r:4.2f}:  acc {acc:6.2f}%   caduta {drop:6.2f}"
                  f"{flag}", flush=True)

    # Summary: the largest rate whose accuracy drop stays within a budget.
    print("\n" + "=" * 78, flush=True)
    print("RIASSUNTO: massimo tasso di potatura entro una data caduta di "
          "accuratezza", flush=True)
    print("=" * 78, flush=True)
    hdr = f"{'layer':>22}{'DC':>7}" + "".join(f"{f'<{b}pt':>8}" for b in (1, 2, 5))
    print(hdr, flush=True)
    for li, (name, p) in enumerate(layers):
        cells = ""
        for budget in (1.0, 2.0, 5.0):
            ok = [r for (l, _, _, r, _, d) in rows if l == li and d <= budget]
            cells += f"{max(ok):>8.2f}" if ok else f"{'-':>8}"
        print(f"{name:>22}{dc_rates[li]:>7.2f}{cells}", flush=True)

    if args.csv:
        with open(args.csv, "w") as f:
            f.write("layer_idx,layer_name,numel,rate,accuracy,drop\n")
            for r in rows:
                f.write(",".join(str(x) for x in r) + "\n")
        print(f"\nscritto {args.csv}", flush=True)


if __name__ == "__main__":
    main()
