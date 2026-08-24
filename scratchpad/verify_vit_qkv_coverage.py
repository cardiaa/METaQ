"""Check that the quantized-tensor predicate covers ViT-B/16's packed QKV weights.

Runs on CPU in a couple of seconds, no checkpoint download: the predicate only
looks at parameter names and shapes, so randomly initialised models are enough.

Background. torchvision's ViT builds its attention out of nn.MultiheadAttention,
which stores the packed query/key/value projection in a bare Parameter named
`in_proj_weight`. The selection rule in utils/trainer_on_gpus_pretrained.py used
to require a ".weight" suffix, so those twelve tensors were never quantized and
every reported ratio was taken against 75.15% of the model. Runs test_250 to
test_254 are affected; DeiT and both ResNets are not.

Expected output after the fix:

    ResNet-18       21 tensors   99.909% of parameters
    ResNet-50       54 tensors   99.788% of parameters
    ViT-B-16        50 tensors   99.682% of parameters
"""

import torch
import torchvision.models as tvm


def quant_param_indices(named_params):
    """The predicate as it stands in the trainer."""
    return [
        idx
        for idx, (name, param) in enumerate(named_params)
        if (name.endswith(".weight") or name.endswith("in_proj_weight"))
        and param.ndim in (2, 4)
    ]


def report(label, model):
    named = list(model.named_parameters())
    idxs = quant_param_indices(named)
    total = sum(p.numel() for _, p in named)
    quant = sum(named[i][1].numel() for i in idxs)
    print(
        f"{label:<14} {len(idxs):>3} tensors   "
        f"{quant / total:.3%} of parameters   ({quant}/{total})"
    )
    return named, idxs


def main():
    torch.manual_seed(0)

    report("ResNet-18", tvm.resnet18(weights=None))
    report("ResNet-50", tvm.resnet50(weights=None))
    named, idxs = report("ViT-B-16", tvm.vit_b_16(weights=None))

    picked = {named[i][0] for i in idxs}
    qkv = [name for name, _ in named if name.endswith("in_proj_weight")]
    assert len(qkv) == 12, f"expected 12 packed QKV tensors, found {len(qkv)}"
    missing = [name for name in qkv if name not in picked]
    assert not missing, f"packed QKV weights still skipped: {missing}"

    # The things that must stay in floating point on a ViT: the class token, the
    # positional embedding, every bias and every LayerNorm vector.
    for name, param in named:
        if param.ndim == 1 or name.endswith("class_token"):
            assert name not in picked, f"{name} must not be quantized"
    assert "encoder.pos_embedding" not in picked, "pos_embedding must stay FP32"

    print("\nOK: packed QKV covered, embeddings and 1-D tensors still excluded.")


if __name__ == "__main__":
    main()
