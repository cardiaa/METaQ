#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_ObCTDoNN
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=16:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# test_257: ResNet-18 at T3=1e-7 with one hundred epochs. ONE flag differs from
# test_238: --n_epochs 60 -> 100.
#
# WHY THIS RUN EXISTS. A literature sweep on 24 August 2026 turned up the one
# competitor that is currently ahead of us in our own headline cell. Oktay et
# al., "Scalable Model Compression by Entropy Penalized Reparameterization",
# ICLR 2020, Table 1: ResNet-18 on ImageNet, 46.7MB uncompressed at 30.0 per
# cent top-1 error, coded with their DFT variant to 1.97MB, 24x, at 30.0 per
# cent error. Lossless at 24x on entropy-coded size, which is exactly our metric
# and exactly our claim. Their SQ variant is 2.78MB, 17x, also lossless.
#
# WHERE WE STAND AGAINST IT, on the same axes and against 69.734:
#   test_239/247/248  T3=6e-8   60 ep   70.020 +0.286   4.80%   20.83x
#   test_238          T3=1e-7   60 ep   69.606 -0.128   4.14%   24.15x
# At their ratio we are 0.128 under the checkpoint where they report parity, so
# at 24x they are marginally ahead. Our lossless point is at 20.8x, theirs at
# 24x. This is the only cell of the paper where we are not in front.
#
# WHY BUDGET AND NOT DOSE. test_238 is 0.098 points from the checkpoint and its
# last three epochs read 69.548, 69.636, 69.606, so the accuracy has flattened
# but the packet has not: 40 to 60 epochs bought +0.200 of accuracy AND -0.07 of
# packet at the same time. Every ResNet-18 run in this sequence has ended with
# both axes still moving, and the seed study puts the noise at 0.083, so 0.098
# is about one standard deviation away. Raising the budget is the manoeuvre that
# has never once cost us the other axis; raising or lowering T3 trades them.
# The dose model also says the zero crossing at 60 epochs is near T3=8e-8, which
# would land at roughly 22.5x, still short of 24x. Budget is the only lever that
# can beat them on BOTH axes rather than on one.
#
# WHAT SUCCESS LOOKS LIKE. 69.74 or better at 4.1 per cent or less, that is
# lossless at 24x or beyond, which takes the cell back. Anything at or above
# 69.73 at 4.14 per cent already ties them at their own ratio with a deployable
# four-bit model rather than a reparameterized FP32 one.
#
# WHAT TO SAY IF IT FAILS. If the accuracy plateaus around 69.65 the honest
# reading is that EPR-DFT holds the 24x lossless point on ResNet-18 and we hold
# ResNet-50, DeiT-Small and ViT-B/16, where they publish nothing. That is still
# a paper; it just needs the comparison table to say so plainly. The fallback
# lever, untried on this network, is --layer_C with the input convolution at 8
# bits: 9408 of 11.68M parameters, so about 0.01 per cent of the packet, and it
# was worth +0.28 on AlexNet for a comparable price.
#
# NOTHING ELSE MOVES. T3 stays at 1e-7, T2 at 1e-7, T1 at 1e-5, the dual step at
# 1.3e-2 relative, min_lr at 1e-4.
#
# COST: 447s/epoch measured on this configuration, so about twelve and a half
# hours for one hundred epochs. The sixteen-hour wall is slack, and there is no
# resume in the trainer, so it has to fit in one allocation.

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
    echo "metaq environment not found: $METAQ_PYTHON" > "$LOG_FILE"
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
    --model_name ResNet-18 \
    --delta -10 \
    --gamma 2 \
    --data_root /leonardo_work/IscrC_ObCTDoNN/acardia0/datasets \
    --train_workers 4 \
    --val_workers 2 \
    --batch_size 64 \
    --n_epochs 100 \
    --lr 1e-2 \
    --optimizer_weight_decay 1e-4 \
    --perspective_coeff 1e-5 \
    --entropy_coeff 1e-7 \
    --sparsity_coeff 1e-7 \
    --perspective Y \
    --flat_schedule N \
    --mag_prune_ratio 0 \
    --quantization Y \
    --quantizer lsq \
    --lsq_scale_lr 1e-5 \
    --lsq_scale_lr_schedule Y \
    --min_lr 1e-4 \
    --lsq_init mse \
    --lsq_grad_scaling N \
    --joint_lsq_metaq Y \
    --bn_recalibration_batches 50 \
    --C 16 \
    --max_iterations 3 \
    --metrics_interval 1 \
    --entropy_warmup_epochs 1 \
    --entropy_every 4 \
    --dual_step 1.3e-2 \
    --dual_step_mode relative \
    --check_ddp_sync \
    --pretrained Y \
    > "$OUTPUT_TARGET" 2>&1
'
