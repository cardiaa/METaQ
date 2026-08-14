#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_ObCTDoNN
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=02:30:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# test_196: PEAQ on ResNet-18 with the new schedule tricks, no distillation.
#
# Test_195 showed the DeiT recipe does not transfer to ResNet-18. As a control
# with PEAQ off, LSQ plus distillation at alpha 0.9 reached only 69.25, well
# below the 69.89 that plain LSQ reached without distillation (test_174), and
# below the 69.76 FP32 line. The reason is structural: on DeiT plain LSQ drifts
# down and needs an anchor, whereas on ResNet-18 4-bit LSQ already exceeds full
# precision, so anchoring it to the teacher at a strong alpha drags it back down
# rather than up. A min_lr floor at 5% of base also keeps the rate churning
# where ResNet-18 wants it to die and settle. Both were medicine for a disease
# ResNet-18 does not have.
#
# So distillation is dropped here. The goal for ResNet-18 is not to create
# accuracy margin, which plain LSQ already has, but to push PEAQ past the 5.88%
# that test_168 delivered at 69.72 (essentially lossless). This run is test_168
# exactly, PEAQ on with the validated coefficients and no distillation, plus the
# two schedule improvements developed on DeiT that do not depend on an anchor:
# the step-size schedule, so the LSQ grid decays with the weights instead of
# drifting at a constant rate, and a cosine floor, here at 1% of base (1e-4)
# rather than DeiT's 5%, since ResNet-18 benefits from the rate dying for
# accuracy while still leaving PEAQ enough life in the tail to keep compressing.
#
# WHAT TO READ. Sparse accuracy against the 69.76 FP32 line and test_168's
# 69.72; sparse ratio against test_168's 5.88%. If the tail improvements let
# PEAQ compress below 5.88% at unchanged accuracy, the new machine beats the old
# ResNet-18 result. If min_lr at 1% still churns too much, accuracy will sit
# below 69.7 and the floor should drop further.

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
    --n_epochs 20 \
    --lr 1e-2 \
    --optimizer_weight_decay 1e-4 \
    --perspective_coeff 1e-5 \
    --entropy_coeff 3e-8 \
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
    --dual_step 3e-9 \
    --check_ddp_sync \
    --pretrained Y \
    > "$OUTPUT_TARGET" 2>&1
'
