#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_ObCTDoNN
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=1:30:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# test_182: does the rise at lsq_scale_lr = 1e-6 continue, or does it flatten?
#
# The step size a of a tensor fixes where its sixteen representable levels sit,
# and lsq_scale_lr is how fast those levels are allowed to move. Sweeping it
# over four decades with plain LSQ, ten epochs each, and reading the level
# reached over the last three epochs, which unlike a trend does not depend on
# where a run happens to start:
#
#     1e-4  78.959   scales moved 10.5%    zstd 10.64 -> 10.94
#     1e-5  79.014                         zstd 10.67 -> 10.91
#     1e-6  79.145   scales moved  3.8%    zstd 10.71 -> 10.79
#     1e-7  79.108   scales moved  0.53%   zstd 10.74 -> 10.74
#
# Slower is better up to an optimum near 1e-6, past which freezing the scales
# outright costs a little. Slow scales also preserve compressibility, which
# degrades steadily at the faster rates. The 1e-5 we had used since ResNet-18,
# and which every PEAQ run so far inherited, is on the wrong side of that
# optimum.
#
# The gain over 1e-5 is only 0.13 though, a fifth of the 0.70 that separates us
# from the 79.742 FP32 baseline. What justifies this longer run is one specific
# observation: at epoch 10 the 1e-6 run was still climbing (79.090, 79.138,
# 79.208) while the 1e-7 run had already flattened (79.104, 79.104, 79.116).
# Twenty-five epochs settle whether that climb is real. Reaching 79.4 or above
# changes the picture for DeiT; flattening near 79.15 closes this line with a
# modest but adoptable improvement.
#
# Everything else is test_173: plain LSQ, per-tensor, MSE initialization, four
# bits on all fifty tensors, no regularizer.
#
# Twenty-five epochs at about 143s each: roughly one hour.
#
# The configuration matches test_171 exactly except that all three coefficients
# are zero. That combination is what actually disables the regularizer: the
# closed-form branch guarding the perspective ridge and the sparsity term
# requires one of them to be nonzero, and the entropy branch requires a positive
# entropy coefficient, so neither contributes a gradient. Note that
# entropy_warmup_epochs alone would NOT have been enough, since it postpones the
# entropy coefficient only and leaves the other two terms active, which is why
# the first epoch of tests 169 to 171 was not a plain-LSQ measurement. The
# perspective flag itself stays Y because joint LSQ-PRESTO requires it and
# because turning it off would enable the legacy non-perspective dual instead.
#
# Reading against the 78.844 that test_171 reached at epoch 18: a plain-LSQ
# result near 79.7 means the missing accuracy is the PRESTO dose and the fix is
# to retune it, whereas a plain-LSQ result also near 78.8 means the limit is in
# the quantization setup and per-channel step sizes become the way forward.
#
# Without the regularizer an epoch costs about 133s instead of 420s, so the
# whole run takes roughly 45 minutes.

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
    --lr 2e-5 \
    --optimizer_weight_decay 0 \
    --perspective_coeff 0 \
    --entropy_coeff 0 \
    --sparsity_coeff 0 \
    --perspective Y \
    --flat_schedule N \
    --mag_prune_ratio 0 \
    --quantization Y \
    --quantizer lsq \
    --lsq_scale_lr 1e-6 \
    --lsq_init mse \
    --lsq_grad_scaling N \
    --joint_lsq_metaq Y \
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
