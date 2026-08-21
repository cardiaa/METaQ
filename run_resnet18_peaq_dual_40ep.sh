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

# test_236: ResNet-18, the test_235 configuration with twice the budget. ONE
# flag differs from test_235: --n_epochs 20 -> 40.
#
# WHY. Both axes of test_235 were still moving when its cosine ran out. Over its
# last five epochs the accuracy rose at +0.134 per epoch and the packet fell at
# -0.032 per epoch, so the run was stopped by the budget and not by the method.
# That is the regime in which epochs are the cheapest knob available, because
# they buy BOTH axes at once: raising T3 would buy size and spend accuracy,
# which is the wrong trade when accuracy is what binds.
#
# WHERE THE THREE RESNET-18 RUNS STAND, all against 69.734 FP32:
#
#   test_174   LSQ only,          20 ep   69.888  +0.154   9.34%   10.71x
#   test_196   METaQ, slow dual,  20 ep   69.840  +0.106   6.08%   16.45x
#   test_235   METaQ, fast dual,  20 ep   69.668  -0.066   5.74%   17.42x
#   test_197   METaQ, slow dual,  40 ep   69.578  -0.156   5.21%   19.19x
#
# test_235 bought 5.6% of relative size over test_196 for 0.172 accuracy points,
# which is smaller than the 0.338 that separates the two runs at epoch one, where
# the entropy term is still switched off and the two are functionally identical.
# The gap is therefore inside the run-to-run noise of the pair, but its central
# value sits below FP32, and that is not a row worth defending.
#
# WHAT WOULD MAKE THIS RUN THE HEADLINE. test_197 shows what forty epochs bought
# with the slow dual: 5.21%, but at -0.156. test_235 shows what the fast dual
# bought at twenty: the same accuracy region with 0.34 points less packet. If
# the two effects simply add, this run lands near 5.2% with the accuracy still
# climbing through the last epochs, and 19x at or above FP32 becomes the row.
#
# WHAT WOULD MAKE IT A NULL. The accuracy plateaus early and the extra epochs
# only buy size, ending below FP32 again. Then the honest headline stays
# test_196 at 16.45x, and test_235 is reported as the frontier point next to it.
#
# UNCHANGED FROM test_235, deliberately: T3 stays at 3e-8 and the dual step
# stays at 1.3e-2 relative. Two knobs moved at once cannot be read.
#
# COST: 432s/epoch measured on test_235, so about five hours for forty epochs.
# The eight-hour wall is slack; there is no resume in the trainer.

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
    --n_epochs 40 \
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
    --dual_step 1.3e-2 \
    --dual_step_mode relative \
    --check_ddp_sync \
    --pretrained Y \
    > "$OUTPUT_TARGET" 2>&1
'
