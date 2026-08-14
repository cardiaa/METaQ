#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_ObCTDoNN
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=06:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# test_197: push ResNet-18 down its frontier, then let it heal.
#
# Test_196 settled the schedule question for ResNet-18: the scale schedule and
# min_lr, useful on DeiT, do not transfer. With min_lr at 1% equal to test_168's
# implicit floor, the only real change was the step-size schedule, and it left
# ResNet-18 slightly worse, 69.70 at 6.10% against test_168's 69.72 at 5.88%.
# ResNet-18 does not have the disease those tools cure: plain LSQ already exceeds
# full precision here, and the network heals during training rather than
# drifting. So test_168 stands as the ResNet-18 result, 5.88% at 69.72, about
# 17x and essentially lossless, already beating L2's 12 to 15x at -0.3 to -0.9.
#
# The opportunity is not a new machine but the frontier. Test_168 sits at only
# -0.04, so there is roughly 0.8 of accuracy margin before we reach the -0.86
# that L2 accepts at 15x. Spending some of that margin on compression should give
# a more dominant near-lossless point, exactly as mapping the entropy dose did on
# DeiT.
#
# This run doubles entropy_coeff to 6e-8 and, crucially, extends the budget the
# RIGHT way. A naive 40-epoch run would stretch the cosine over 40 epochs, the
# artefact that fooled us four times on DeiT. Instead lr_decay_epochs 19 holds
# the descent identical to test_168, floored by epoch 20, and adds 20 epochs at
# the floor. The descent phase is therefore comparable to test_168 at double the
# dose, and the floor tail gives ResNet-18 the epochs to heal. On DeiT that floor
# tail did not recover accuracy (test_193) because DeiT drifts; ResNet-18 heals,
# so this also tests whether the tail works on a network that actually recovers.
# No scale schedule, since test_196 showed it hurts here; the 1% floor is
# test_168's own floor.
#
# Both budget and dose change, so a good landing is attributed to the pair, not
# to one lever. The risk is over-compression: 6e-8 is double the validated dose.
# ResNet-18's healing should absorb it, but if accuracy collapses the frontier is
# steeper than expected and the intermediate dose 4.5e-8 is the fallback.
#
# WHAT TO READ. Sparse accuracy against 69.72 and the 69.76 FP32 line, as a mean
# over the last three epochs; sparse ratio against 5.88%. A landing below 5.88%
# while holding near 69.5 is a stronger near-lossless point than test_168. Watch
# the floor tail, epochs 21 to 40: does accuracy climb there as ResNet-18 heals,
# or flatten as on DeiT?

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

    torchrun \
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
    --n_epochs 40 \
    --lr_decay_epochs 19 \
    --lr 1e-2 \
    --optimizer_weight_decay 1e-4 \
    --perspective_coeff 1e-5 \
    --entropy_coeff 6e-8 \
    --sparsity_coeff 1e-7 \
    --perspective Y \
    --flat_schedule N \
    --mag_prune_ratio 0 \
    --quantization Y \
    --quantizer lsq \
    --lsq_scale_lr 1e-5 \
    --lsq_init mse \
    --lsq_grad_scaling N \
    --joint_lsq_metaq Y \
    --bn_recalibration_batches 50 \
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
