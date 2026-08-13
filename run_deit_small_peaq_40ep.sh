#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_ObCTDoNN
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=6:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# test_192: give PEAQ the epochs to recover, not just to compress.
#
# Two PEAQ runs on the lossless baseline landed at the same place: 79.16 at
# 7.55% (test_190, full dose) and 79.24 at 7.78% (test_191, sparsity halved).
# Halving sparsity_coeff barely moved anything, because sparsity turned out to
# be EMERGENT here, driven to ~44% by the ridge and the entropy term rather than
# by the sparsity coefficient, which is a weak lever at this operating point.
# Both runs sit at the same point on the accuracy-ratio frontier, about 0.5
# below the 79.742 lossless line.
#
# The reason to extend rather than retune is in the trajectory. In both runs
# accuracy dips to about 78.5 by epoch ten, then climbs monotonically to the end
# and is STILL climbing at epoch twenty-five, while sparsity reaches its ~44%
# plateau already by epoch fifteen. So the compression finishes early and the
# remaining epochs are recovery that has not completed when the run stops.
# Twenty-five epochs are enough to compress but not to compress AND heal. With
# min_lr holding the rate at 5% of base, PEAQ keeps the compression alive through
# the tail instead of releasing it as in test_186, so more epochs should recover
# accuracy while the ratio holds, potentially lossless AND at 7.7% at once.
#
# Only n_epochs changes, 25 to 40. Everything else is test_190: distillation at
# alpha 0.9, lr 1e-4, min_lr 5e-6, the step-size schedule, coefficients 1e-5 /
# 3e-8 / 1e-7. sparsity_coeff is kept at the standard 1e-7 since test_191 showed
# it barely matters.
#
# HONEST CAVEAT. A late climb has fooled us three times as a cosine-tail
# artefact (tests 178, 179, 182). The difference claimed here is mechanistic:
# there the climb was the rate dying and the model freezing, whereas here the
# compression genuinely completes early and the rest is healing with the rate
# still alive at the 5% floor. That is an inference, and this run is its test. If
# accuracy clears 79.5 the recovery is real and we may reach lossless at maximum
# compression; if it flattens near 79.25 it was the tail again and the frontier
# is what it is, at which point lowering entropy_coeff maps the trade instead.
#
# WHAT TO READ. Sparse accuracy against 79.24 and the 79.742 lossless line, as a
# mean over the last three epochs; sparse ratio against 7.78%, which should HOLD
# rather than drift up; and whether accuracy is still rising at epoch 40 or has
# plateaued.
#
# Forty epochs at about 430s each with the entropy solver: roughly four and a
# half hours.

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
    --n_epochs 40 \
    --lr 1e-4 \
    --optimizer_weight_decay 0 \
    --perspective_coeff 1e-5 \
    --entropy_coeff 3e-8 \
    --sparsity_coeff 1e-7 \
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
