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

# test_188: a faster weight rate, now that the constraint that set it is gone.
#
# Test_187 closed the quantizer question for good. Reviving per-channel step
# sizes and eight-bit ends on top of distillation gave 79.569 over its last
# three epochs against the 79.585 the plain configuration reaches at the same
# epoch, so nothing, while costing 0.65 points of sparse ratio, 11.77% against
# 11.12%. The prediction had been 79.93. Those levers had looked positive in the
# drifting regime because capacity was then the binding constraint; with the
# network anchored to its teacher it no longer is. Both are dropped here.
#
# What has never been retried is the weight learning rate. It was fixed at 2e-5
# in tests 169 to 171 for one reason only, to MINIMISE DRIFT DAMAGE: at 1e-4 the
# network wandered off, ending test_170 at 77.85. That constraint no longer
# exists, drift being exactly what distillation removed, so we may simply have
# been undertraining ever since. Q-ViT uses a base rate of 2e-4, and under their
# linear batch scaling the equivalent for our global batch of 2048 would be
# 8e-4, forty times ours. That figure is not transferable, since LAMB exists
# precisely to make such rates survivable where Adam does not, but it shows how
# far from the usual regime we sit. This run takes the cautious step to 5e-5,
# with 1e-4 to follow if it pays.
#
# The step-size schedule fix rides along deliberately. Their own optimizer had
# decoupled the step sizes from the weight schedule: weights decay to 1% of base
# while the step sizes kept a constant 1e-5, so their relative speed went from
# 0.5x at epoch one to 35x at epoch twenty-five, and a grid still moving after
# the network has frozen keeps reassigning weights between levels. That is not a
# tuning knob but an alignment with canonical LSQ and with Q-ViT, where alpha
# lives in the weight optimizer and decays by construction. Sweeping the rate
# without it would measure a regime we are about to abandon. If this run
# disappoints, the two changes have to be separated.
#
# Twenty-five epochs are kept so the cosine has exactly the shape it has in
# test_185. Comparing across schedule lengths has misled us three times: the
# late climb of test_179, the apparent ceiling of test_178 and the compression
# plateau of test_186 were all schedule artefacts.
#
# WHAT TO READ, against the 79.592 and 11.09% test_185 reaches over its last
# three epochs: the sparse accuracy, which is the delivered metric, and whether
# the epoch-to-epoch jitter shrinks, which is what the step-size fix should buy.
# Lossless target: 79.742 as a mean over the last three epochs, not as a peak,
# since with a spread near 0.1 the best of fifteen epochs sits about 0.15 above
# the mean by chance alone.
#
# About an hour.

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
    --lr 5e-5 \
    --optimizer_weight_decay 0 \
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
    --lsq_scale_lr_schedule Y \
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
