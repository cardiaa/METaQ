#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_ObCTDoNN
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=01:30:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# test_195: ResNet-18 control with the new pipeline, Phase 3.
#
# The DeiT machine is now settled: distillation from a frozen full-precision
# teacher, a step-size schedule that decays with the weights, and a min_lr floor
# that keeps PEAQ alive through the tail. This carries that pipeline to
# ResNet-18, whose earlier results (test_168 at 5.88%, test_174 control at
# 69.888) predate all of it and are no longer the right comparison.
#
# ResNet-18 is a different regime from DeiT and does NOT need the low learning
# rate. It never drifted: plain LSQ healed from 67 to 69.888, ABOVE its own
# 69.734 checkpoint (test_174). So the learning rate stays at 1e-2, as in
# test_168 and test_174, and only the pipeline additions are folded in:
# distillation at alpha 0.9, the step-size schedule, and a min_lr floor at 5e-4,
# which is 5% of the base rate as on DeiT. The regularizer stays OFF; this is the
# matched lossless-baseline control, and PEAQ follows in the next run.
#
# The purpose of distillation here is not to reach lossless, which plain LSQ
# already clears, but to create accuracy HEADROOM above the FP checkpoint that
# PEAQ can later convert into compression. On DeiT distillation lifted the
# baseline by about 0.5; if it lifts ResNet-18 above 69.888, that margin is
# convertible to ratio, which is the metric ResNet-18 sells, since L2 reports
# 12x at -0.27 and 15x at -0.86 while test_168 already reached 17x at -0.01.
#
# Everything else is test_174: lr 1e-2, batch 64, 20 epochs, weight decay 1e-4,
# BN recalibration on 50 batches, C=16, coefficients zero.
#
# WHAT TO READ. Sparse accuracy against test_174's 69.888 and the 69.734 FP
# checkpoint: how much headroom distillation buys. Without the entropy solver an
# epoch is cheap, so twenty epochs run in well under an hour.

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
    --n_epochs 20 \
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
    --lsq_init mse \
    --lsq_grad_scaling N \
    --joint_lsq_metaq Y \
    --lsq_scale_lr_schedule Y \
    --min_lr 5e-4 \
    --distillation Y \
    --distill_alpha 0.9 \
    --distill_tau 1.0 \
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
