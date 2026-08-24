#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_ObCTDoNN
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=05:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# test_255: ViT-B/16, LSQ-only control at twenty epochs, rerun of test_251 on
# the corrected tensor set. Nothing about the recipe changes; what changes is
# what the recipe is applied to.
#
# WHY IT HAS TO BE RERUN. torchvision builds ViT attention out of
# nn.MultiheadAttention, which keeps the packed query/key/value projection in a
# bare Parameter named in_proj_weight instead of a Linear submodule. The
# selection rule in the trainer required a ".weight" suffix, so those twelve
# tensors, 21.2M parameters, were never quantized and never counted. Runs 250 to
# 254 all print
#   [QUANTIZED TENSORS] count=38, parameters=65058816, fraction_of_model=75.15%
# and every ratio in them is taken against that subset, not against the model.
# Every other network we have run is above 99 per cent, which is why this went
# unnoticed for five runs. The predicate now also accepts in_proj_weight and the
# header should read count=50, parameters=86292480, fraction_of_model=99.68%.
# Check that line before reading anything else in the log.
#
# WHAT THE FIVE CONTROLS SAID, on the 75 per cent accounting. FP32 is 81.056.
#   250  10ep lr 5e-5  80.814 @ 10.13%
#   251  20ep lr 5e-5  80.914 @ 10.26%   best 80.975 at epoch 18
#   252  20ep lr 1e-4  80.830 @ 10.28%
#   253  20ep lr 3e-5  80.874 @ 10.24%
#   254  30ep lr 5e-5  80.833 @ 10.35%   best 80.921 at epoch 28
# The rate is not the binding constraint: three rates within a factor of three
# land within 0.09 of each other, and thirty epochs are worse than twenty. Nor
# is the network frozen, level_change runs 10.5% down to 1.1%. Four-bit LSQ
# simply sits about a tenth of a point under the checkpoint on this network and
# stays there, so 251 is the recipe and twenty epochs is the budget.
#
# THE HEADROOM IS THE THING TO WATCH. On both ResNets the LSQ control finishes
# ABOVE full precision, +0.756 and +0.432, and METaQ spends that. Here it
# finishes at -0.14, so ViT-B/16 belongs with DeiT-Small: report the frontier
# against the budget-matched control, not a lossless claim against FP32.
#
# WHAT WE ARE SHOOTING AT. NNCodec, Table 1 of the ICML 2023 workshop paper,
# reports ViT-B/16 at 346.27MB and 81.07 top-1, coded to 32.87MB at -1.12, that
# is 9.49 per cent of the model. Their Figure 7 gives the whole grid; the
# Pareto-best rows are 80.92 at 14.94%, 80.81 at 12.10%, 80.41 at 10.44% and
# 79.95 at 9.49%. On honest full-model accounting test_251 would already be
# near that curve, which is why the accounting has to be right before any of
# this goes in a table.
#
# THE CHECKPOINT MUST BE STAGED FIRST. From a login node:
#   wget -P /leonardo_work/IscrC_ObCTDoNN/acardia0/imagenet_checkpoints \
#        https://download.pytorch.org/models/vit_b_16-c867db91.pth
#
# COST: measured at 198s per epoch on runs 250-254 at 75 per cent coverage.
# The extra 21M parameters are quantization work, not forward work, so expect
# 210-230s and about eighty minutes for twenty epochs. The five-hour wall is
# slack.

LOG_DIR=$WORK/acardia0/LeonardoTests
mkdir -p "$LOG_DIR"

LAST=$(find "$LOG_DIR" -maxdepth 1 -type f -name 'Leonardo_test_*.log' -printf '%f\n' | grep -oE '^Leonardo_test_[0-9]+\.log$' | grep -oE '[0-9]+' | sort -n | tail -1)

if [ -z "$LAST" ]; then
    NEXT=1
else
    NEXT=$((LAST+1))
fi

LOG_FILE=$LOG_DIR/Leonardo_test_${NEXT}.log
export LOG_FILE

module load profile/deeplrn
module load cineca-ai/4.3.0

METAQ_PYTHON=$WORK/acardia0/venvs/metaq/bin/python
if [ ! -x "$METAQ_PYTHON" ]; then
    echo "DeiT-Small environment not found: $METAQ_PYTHON" > "$LOG_FILE"
    exit 1
fi
export METAQ_PYTHON

export OMP_NUM_THREADS=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export PYTHONWARNINGS="ignore::UserWarning"
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

cd "$WORK/acardia0/METaQ"

srun --ntasks=$SLURM_NTASKS --ntasks-per-node=1 bash -lc '
    if [ "$SLURM_NODEID" -eq 0 ]; then
        OUTPUT_TARGET="$LOG_FILE"
    else
        OUTPUT_TARGET=/dev/null
    fi

    "$METAQ_PYTHON" -m torch.distributed.run \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=4 \
    --node_rank=$SLURM_NODEID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --rdzv_id=$SLURM_JOB_ID \
    train_on_gpus_pretrained.py \
    --model_name ViT-B-16 \
    --delta -10 \
    --gamma 2 \
    --data_root /leonardo_work/IscrC_ObCTDoNN/acardia0/datasets \
    --train_workers 4 \
    --val_workers 2 \
    --batch_size 64 \
    --n_epochs 20 \
    --lr 5e-5 \
    --optimizer_weight_decay 0 \
    --perspective_coeff 0 \
    --entropy_coeff 0 \
    --sparsity_coeff 0 \
    --perspective Y \
    --flat_schedule N \
    --mag_prune_ratio 0 \
    --quantization Y \
    --quantizer lsq \
    --lsq_scale_lr 1e-5 \
    --lsq_init mse \
    --lsq_grad_scaling N \
    --lsq_scale_lr_schedule Y \
    --min_lr 2.5e-6 \
    --joint_lsq_metaq Y \
    --distillation Y \
    --distill_alpha 0.9 \
    --distill_tau 1.0 \
    --bn_recalibration_batches 0 \
    --C 16 \
    --max_iterations 6 \
    --metrics_interval 1 \
    --entropy_warmup_epochs 1 \
    --entropy_every 4 \
    --dual_step 1.3e-2 \
    --dual_step_mode relative \
    --check_ddp_sync \
    --pretrained Y \
    > "$OUTPUT_TARGET" 2>&1
'
