#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_ObCTDoNN
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# test_250: ResNet-50, the frontier point below lossless. ONE flag differs from
# test_243: --entropy_coeff 3e-8 -> 1e-7.
#
# WHERE ResNet-50 STANDS, against a 76.122 checkpoint:
#
#   test_242   LSQ only, 20 ep    76.554  +0.432   9.78%   10.22x
#   test_243   T3=3e-8,  20 ep    76.318  +0.196   4.92%   20.33x   <- lossless
#
# test_243 halves the control's packet for 0.236 accuracy points and still ends
# above the checkpoint, with the accuracy climbing at +0.197 per epoch over its
# last five. It left 0.196 points of margin unspent, and this run spends them.
#
# WHY 1e-7. It is the same jump that traced the ResNet-18 frontier, where 3e-8 to
# 1e-7 at a matched budget cost 0.674 accuracy points and bought 1.52 points of
# packet. Carried over, this lands near 3.7% of FP32, about 27x, at roughly half
# a point below the checkpoint. That is the second point the cell needs: one row
# at parity and one past it, so the paper reports a frontier on ResNet-50 as it
# does on DeiT-Small rather than a single dot.
#
# WHAT IT IS WORTH EVEN IF IT OVERSHOOTS. The two published methods that measure
# size after lossless coding on this network, DeepCABAC at 9.86x and HEMP at
# 11.26x, both sit 1.6 points below their baseline. A row at 27x and half a point
# down beats them on both axes by a wide margin, and a row that lands worse still
# fixes the slope of the frontier between 20x and whatever it reaches.
#
# COST: 5.5 hours measured on test_243 at the same budget. The dual solver on
# this network costs about 2.7s per call against 0.93s on ResNet-18, which
# follows the tensor count (54 against 21) rather than the weight count: the
# solver is dominated by per-tensor overhead, not by the per-weight sweep.
#
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
    --model_name ResNet-50 \
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
    --entropy_coeff 1e-7 \
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
    --dual_step 1.3e-2 \
    --dual_step_mode relative \
    --check_ddp_sync \
    --pretrained Y \
    > "$OUTPUT_TARGET" 2>&1
'
