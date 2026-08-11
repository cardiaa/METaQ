"""Single-process CPU smoke test of the per-channel path through the REAL trainer.

Offline unit checks cannot catch shape bugs that live in the trainer's own
plumbing: test_176 died on a second, separate construction of v_list that the
unit tests never touched. This runs train_and_evaluate itself, on a tiny
transformer-shaped model and a handful of synthetic batches, with a one-rank
gloo process group so every dist call is real. It exercises the warm-up epoch
(closed-form ridge) and at least one post-warm-up epoch (entropy FISTA), plus
the evaluation and serialization path.

Usage:  python3 scratchpad/smoke_per_channel.py [per_channel|per_tensor|both]
"""
import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.trainer_on_gpus_pretrained import train_and_evaluate  # noqa: E402


class Tiny(nn.Module):
    """Linear stack whose weights have the same rank/shape variety as DeiT:
    non-square tensors, a 4-D conv, and a classifier head."""

    def __init__(self, n_classes=10):
        super().__init__()
        self.stem = nn.Conv2d(3, 16, kernel_size=4, stride=4, bias=False)  # 4-D
        self.qkv = nn.Linear(16, 48, bias=False)                            # wide
        self.proj = nn.Linear(16, 16, bias=False)                           # square
        self.fc1 = nn.Linear(16, 64, bias=False)
        self.fc2 = nn.Linear(64, 16, bias=False)
        self.norm = nn.LayerNorm(16)
        self.head = nn.Linear(16, n_classes, bias=False)

    def forward(self, x):
        h = self.stem(x).flatten(2).mean(dim=2)      # (B, 16)
        a = self.qkv(h)[:, :16] + self.proj(h)
        h = h + a
        h = h + self.fc2(torch.relu(self.fc1(h)))
        return self.head(self.norm(h))


def run(per_channel: bool):
    torch.manual_seed(0)
    device = torch.device("cpu")
    model = Tiny().to(device)
    model = nn.parallel.DistributedDataParallel(model)

    n, img = 64, 8
    x = torch.randn(n, 3, img, img)
    y = torch.randint(0, 10, (n,))
    loader = DataLoader(TensorDataset(x, y), batch_size=16)

    train_and_evaluate(
        model=model,
        model_name="Tiny",
        criterion=nn.CrossEntropyLoss(),
        C=16,
        lr=2e-5,
        lambda_reg=0.0,
        alpha=1.0,
        perspective_coeff=1e-5,
        entropy_coeff=3e-8,
        subgradient_step=1e5,
        w0=0.0,
        r=1.0,
        first_best_indices=20,
        BestQuantization_target_acc=99.8,
        final_target_acc=99.7,
        target_zstd_ratio=0.0179,
        min_xi=0.0,
        max_xi=1.0,
        upper_c=float(sum(p.numel() for p in model.parameters())),
        lower_c=1e-2,
        c1=10,
        c2=1000,
        zeta=50000,
        l=0.5,
        n_epochs=3,                 # warm-up epoch + two entropy epochs
        max_iterations=3,
        device=device,
        train_optimizer="ADAM",
        entropy_optimizer="FISTA",
        trainloader=loader,
        testloader=loader,
        train_sampler=None,
        steps_per_epoch=4,
        delta=-10.0,
        pruning="Y",
        QuantizationType="center",
        sparsity_threshold=1e-3,
        accuracy_tollerance=0.2,
        sparsity_coeff=1e-7,
        use_perspective=True,
        mag_prune_ratio=0.0,
        target_sparsity=0.0,
        metrics_interval=1,
        entropy_warmup_epochs=1,
        entropy_every=1,
        check_ddp_sync=False,
        optimizer_weight_decay=0.0,
        use_quantization=True,
        quantizer="lsq",
        lsq_scale_lr=1e-5,
        lsq_init="mse",
        lsq_grad_scaling=False,
        lsq_per_channel=per_channel,
        joint_lsq_metaq=True,
        bn_recalibration_batches=0,
        dual_step=3e-9,
    )


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29555")
    dist.init_process_group("gloo", rank=0, world_size=1)
    try:
        for mode in (["per_tensor", "per_channel"] if which == "both" else [which]):
            print(f"\n{'=' * 30} {mode} {'=' * 30}", flush=True)
            run(mode == "per_channel")
            print(f">>> {mode}: NESSUN ERRORE", flush=True)
    finally:
        dist.destroy_process_group()
