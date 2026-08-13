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

# test_193: separate compression from healing, as a clean prefix of test_191.
#
# Test_191 reached 79.24 at 7.78% and was still recovering at its last epochs:
# its tail, at a floor rate of 5e-6, gained about 0.05 accuracy per epoch (79.14
# at epoch 22 to 79.29 at epoch 25). It simply ran out of epochs. Test_192 tried
# to help with more budget but STRETCHED the cosine over 40 epochs, which keeps
# the rate high longer, so it compressed harder (48% sparsity by epoch 29 versus
# 44% at epoch 25 in test_191) and pushed recovery even further out. Stretching
# the cosine is the wrong lever: it does more of the same, slower.
#
# This run instead HOLDS the descent length fixed and lengthens only the floor.
# --lr_decay_epochs 24 makes the cosine descend over exactly the 24 epochs it
# used in test_191, reaching the 5e-6 floor at epoch 25, and then n_epochs 40
# holds that floor for fifteen more epochs. Verified offline: epochs 1 to 25 are
# bit-identical to test_191, epochs 26 to 40 sit exactly at 5e-6. So test_191 is
# a literal prefix of this run, and epochs 26 to 40 are a controlled healing
# phase where the cosine no longer moves and PEAQ, whose strength scales with
# the rate, is weak. The question is whether fifteen epochs of floor-rate
# fine-tuning at a fixed compressed configuration recover the roughly 0.5 that
# separates 79.24 from the 79.742 lossless line, while the ratio holds.
#
# This is Andrea's design, and it corrects my earlier mistake of dismissing it
# by pointing at test_192, which tested the opposite thing.
#
# WHAT TO READ. Sparse accuracy across epochs 26 to 40 against 79.24 and the
# 79.742 line: does the ~0.05 per epoch healing seen in test_191's tail continue,
# or does it flatten? Sparse ratio against 7.78%: it should hold or drift only
# slightly, since PEAQ is weak at the floor. If accuracy climbs toward 79.7 while
# the ratio holds near 7.8%, we have lossless at strong compression. If it
# flattens near 79.3, healing is exhausted and the frontier is reached, at which
# point lowering entropy_coeff is the way to trade toward lossless.
#
# Forty epochs, the last fifteen at the floor rate: about four and a half hours.

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
    --lr_decay_epochs 24 \
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
