#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_ObCTDoNN
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# test_240: ResNet-18, LSQ-only control at sixty epochs. The budget-matched
# denominator for test_239, and the last run ResNet--18 needs.
#
# Derived from run_resnet18_peaq_lossless_max.sh by setting the three
# coefficients to zero and changing nothing else, so schedule, codebook, batch,
# warm-up and byte accounting are identical to the row it is the control for.
# It must be at sixty epochs for the same reason the headline is: a sixty-epoch
# cosine at its fortieth epoch is not a forty-epoch cosine at its fortieth, so a
# shorter control cannot stand in for a longer one.
#
# WHY IT MATTERS. The paper currently compares its best rows against test_174, a
# twenty-epoch control that also inherits the test_168 schedule and has no
# --min_lr, so it is mismatched in both budget and schedule. That mismatch is
# written into the limitations; this run removes it.
#
# WHAT IT SETTLES. If it lands near test_174 at roughly 9.3% of FP32, then the
# 35% size reduction reported at twenty epochs holds at sixty and the headline
# margin is real. If plain LSQ also compresses when given a long budget, the
# margin shrinks and the paper must say so. Either way the number goes in the
# table, and ResNet--18 is then closed.
#
# Zeroing all three coefficients is what disables the regularizer: the
# closed-form branch carrying the perspective ridge and the sparsity term
# requires one of them to be nonzero, and the entropy branch requires a positive
# entropy coefficient. --perspective stays Y because joint LSQ-METaQ requires it
# and because turning it off would select the legacy non-perspective dual.
#
# COST: no dual solver, about 140s per epoch, so roughly two hours and twenty.

LOG_DIR=$WORK/acardia0/LeonardoTests
mkdir -p "$LOG_DIR"

LAST=$(find "$LOG_DIR" -maxdepth 1 -type f -name 'Leonardo_test_*.log' -printf '%f\n' | grep -oE '^Leonardo_test_[0-9]+\.log$' | grep -oE '[0-9]+' | sort -n | tail -1)
NEXT=$(( ${LAST:-0} + 1 ))
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
    --n_epochs 60 \
    --lr 1e-2 \
    --optimizer_weight_decay 1e-4 \
    --perspective_coeff 0 \
    --entropy_coeff 0 \
    --sparsity_coeff 0 \
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
