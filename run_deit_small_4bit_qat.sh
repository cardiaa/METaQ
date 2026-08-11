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

# DeiT-Small / ImageNet transfer test for the complete joint LSQ-METaQ method.
# Four 4-GPU nodes give a global batch size of 2048.
#
# test_176 keeps the whole test_171 recipe and changes only the granularity of
# the learned step sizes: one per output channel instead of one per tensor.
#
# The controls have located the missing accuracy exactly. Against an FP32
# baseline of 79.742, plain LSQ converges to 79.04 (test_173) and METaQ on top
# reaches 78.844 (test_171), so the gap splits into 0.70 charged to the
# quantizer and 0.21 charged to the regularizer. On ResNet-18 plain LSQ instead
# reaches 69.888, ABOVE its own 69.734 checkpoint (test_174): per-tensor step
# sizes suffice for the convolutional network and fail for the transformer.
# Test_175 then showed the deficit is not concentrated, since 256 levels on the
# patch embedding and the head bought only 0.10 of the 0.70 while costing 0.16
# of compression ratio. What remains is the granularity of the step size itself.
#
# The METaQ solver is unchanged in cost. Scaling the abscissa by a positive
# constant leaves the lower convex envelope's vertex set untouched, and the
# facet intercept s does not depend on the step size at all, so the envelope is
# built once on the integer codebook and only the slope is rescaled per weight.
# The dual is untouched as well: the counts are counts of INTEGER symbols over
# the tensor, which is exactly what the entropy coder emits.
#
# The extra step sizes are paid for in the reported ratio: 42856 channels across
# the 50 tensors, that is 0.194% of the FP32 weight storage.
#
# FIRST GPU RUN OF THIS CODE PATH. It is verified offline only
# (scratchpad/verify_per_channel_*.py). Check the first epoch before trusting
# the rest: no NaN, xi_pinned_frac at zero, and the per-tensor mean step sizes
# printed by lsq_diag in the same 1e-2 range as test_171.
#
# Reading: epoch 1 runs with the entropy solver off and is directly comparable
# to the 78.908 of test_171 at the same point. At or above 79.4 the granularity
# was the blocker and lossless is in reach; near 78.9 it was not, and the cause
# lies outside the quantizer.

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
    --n_epochs 20 \
    --lr 2e-5 \
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
    --lsq_per_channel Y \
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
