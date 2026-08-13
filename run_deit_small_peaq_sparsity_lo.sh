#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_ObCTDoNN
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=4:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# test_191: back off the sparsity, which is where the accuracy went.
#
# Test_190 switched PEAQ on over the lossless baseline and compressed hard, from
# 79.79 at 11.43% (test_189) to 79.16 at 7.55% over the last three epochs, so
# 2.42 bits per weight. That is our best DeiT ratio yet and already beats Q-ViT
# 3-bit on both axes, 79.16 against 79.0 and 7.55% against 9.86% nominal, but it
# is no longer lossless: the 0.05 margin above the FP32 checkpoint could not
# absorb PEAQ's roughly 0.6-point cost.
#
# The byte breakdown says clearly where to push. In test_190 the value stream,
# the non-zero weights, fell from 9.58% to 4.50%, halved, which is the entropy
# term doing exactly its job. The mask instead GREW from 1.84% to 3.05%, because
# sparsity climbed to 45.85% and that many zeros are expensive to index. So the
# excess cost sits on the sparsity side twice over: it inflates the mask, and
# 46% pruning on a vision transformer almost certainly explains the accuracy
# drop. Lowering the entropy coefficient would have thrown away the part that
# works; the coefficient to move is sparsity.
#
# Only sparsity_coeff changes, halved from 1e-7 to 5e-8, with perspective at
# 1e-5 and entropy at 3e-8 held. Everything else is test_190, including the lr
# floor and the step-size schedule. This is the first bisection point of the
# accuracy-versus-ratio frontier, taken on the coefficient the data implicate.
#
# EXPECTATION, falsifiable. Fewer zeros should recover accuracy toward the
# 79.742 lossless line and lighten the mask, while the value stream stays
# compressed since entropy is untouched. If sparsity settles near 35%, a landing
# around 79.5 to 79.6 at roughly 7.5% is plausible, near-lossless without losing
# ratio, because the ratio the mask was wasting comes back. If accuracy barely
# moves, the cost was not the sparsity after all and the next step lowers
# entropy instead.
#
# WHAT TO READ. Sparse accuracy against 79.16 and the lossless 79.742; sparse
# ratio against 7.55%; and the mask-versus-values split in sparse_zstd_components,
# to confirm the mask shrinks while the values hold.
#
# About three hours with the entropy solver.

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
    --model_name DeiT-Small \
    --delta -10 \
    --gamma 2 \
    --data_root /leonardo_work/IscrC_ObCTDoNN/acardia0/datasets \
    --train_workers 4 \
    --val_workers 2 \
    --batch_size 128 \
    --n_epochs 25 \
    --lr 1e-4 \
    --optimizer_weight_decay 0 \
    --perspective_coeff 1e-5 \
    --entropy_coeff 3e-8 \
    --sparsity_coeff 5e-8 \
    --perspective Y \
    --flat_schedule N \
    --mag_prune_ratio 0 \
    --quantization Y \
    --quantizer lsq \
    --lsq_scale_lr 1e-5 \
    --lsq_init mse \
    --lsq_grad_scaling N \
    --lsq_scale_lr_schedule Y \
    --min_lr 5e-6 \
    --joint_lsq_metaq Y \
    --distillation Y \
    --distill_alpha 0.9 \
    --distill_tau 1.0 \
    --bn_recalibration_batches 0 \
    --C 16 \
    --max_iterations 3 \
    --metrics_interval 1 \
    --entropy_warmup_epochs 1 \
    --entropy_every 4 \
    --dual_step 3e-9 \
    --check_ddp_sync \
    --pretrained Y \
    > "$OUTPUT_TARGET" 2>&1
'
