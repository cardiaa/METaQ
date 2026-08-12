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

# test_190: PEAQ switched on over the lossless baseline. The compression run.
#
# The baseline search is over. On DeiT-Small, distillation from a frozen
# full-precision teacher plus a weight rate raised to 1e-4 reaches 79.79 as a
# mean over the last three epochs (test_189), above the 79.742 FP32 checkpoint,
# so we finally train a quantized model at no accuracy loss. That is the ground
# the compression engine has to sit on. Everything before was about reaching it:
# the quantizer was never the limit (tests 175 to 182), the missing pieces were
# the anchor and, once the anchor removed the drift, a rate no longer held down
# to limit that drift.
#
# This run turns on the three coefficients at the values validated jointly on
# both architectures, perspective 1e-5, entropy 3e-8, sparsity 1e-7, changing
# nothing else with respect to test_189 except the learning-rate floor. The pair
# 189 to 190 therefore isolates PEAQ exactly at the new, lossless operating
# point.
#
# min_lr = 5e-6 is the lesson of test_186. There the cosine floored at 1% of the
# base rate, so the last eight epochs ran at a rate that had collapsed toward
# zero, and since PEAQ enters by being added to the loss gradient it switched
# off with it: compression bottomed at 8.18% and then REVERSED to 8.27% while
# the network was already frozen. A floor at 5% of the 1e-4 base, the same
# relative floor Q-ViT uses, keeps the engine running through the whole
# schedule. The step-size schedule fix from test_188 rides along so the grid
# decays with the weights rather than drifting at a constant rate.
#
# EXPECTATION, stated to be falsifiable. On the matched pair without
# distillation PEAQ cost about 0.2 accuracy and bought roughly 2 points of
# sparse ratio. Here it starts from 79.79 and 11.43%, so a landing near 79.6 at
# perhaps 9.5% would already be lossless and well below the 12.93% a nominal
# four-bit DeiT occupies. But the balance may differ: distillation makes the
# loss a stronger opponent to PEAQ, so the regularizer may compress less at the
# same dose, or equivalently there may be room to raise entropy later. min_lr
# should also let the tail keep compressing where test_186 stalled.
#
# WHAT TO READ. Sparse accuracy against 79.79, the gap is the cost; sparse ratio
# against 11.43%, the gap is the gain. Watch sparsity: with min_lr on it should
# now KEEP RISING through the last third instead of stalling as in test_186. All
# metrics wanted for the paper are already logged: sparse_accuracy, sparse_ratio,
# H_Q bits per weight, dense zstd, sparsity.
#
# With the entropy solver, about 430s per epoch: twenty-five epochs near three hours.

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
