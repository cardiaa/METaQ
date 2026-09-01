#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_ObCTDoNN
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=1:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# test_183: does distillation stop DeiT from drifting away?
#
# Every DeiT run so far has the same shape: a peak within five to seven epochs,
# a trough, a partial recovery once the cosine kills the learning rate, and a
# landing at 79.0 against an FP32 baseline of 79.742. Seven plain-LSQ runs land
# there regardless of the quantizer: per-channel step sizes (176, 178), the
# canonical initialization (178), eight bits on the first and last layer (175,
# 178) and the step-size learning rate swept over four decades (179 to 182) all
# leave that plateau where it is. The quantizer is not what limits us.
#
# What limits us is that nothing anchors the network. Weight decay is zero and
# the augmentation is RandomResizedCrop plus a horizontal flip, which is the
# ResNet recipe; DeiT was trained with RandAugment, mixup, cutmix and random
# erasing. Given a 22M-parameter transformer and no regularization, training
# does not converge towards a better model, it wanders away from a good one.
# Test_177 made that explicit by losing 1.13 points over fifty epochs.
#
# Distillation is the most direct answer to that specific failure. A frozen copy
# of the pretrained network is kept as a teacher, and at every step the student
# is asked to reproduce its full class distribution rather than merely predict
# the label. The teacher IS the function we are trying to preserve, so this is
# an anchor in the literal sense, and the soft targets carry far more
# information than a one-hot label. Q-ViT reaches 80.9, above full precision,
# largely because of distillation; DeiT itself was designed around it.
#
# The regularizer stays OFF. This asks whether the drift stops, and the entropy
# solver would triple the cost of asking. Only once the answer is known does it
# make sense to switch PEAQ back on and retune its coefficients in the new
# balance. Nothing in the PEAQ derivation is affected either way: its gradient
# is built from the weights and the step sizes and never looks at the loss.
#
# WHAT TO READ, against test_173, which runs 79.056, 79.120, 79.044, 79.110,
# 78.996, 78.984, 78.986, 78.998, 78.976, 79.068 over its first ten epochs:
# does the trajectory stop sagging in the middle? The signature of success is
# the disappearance of the epoch 5 to 7 trough, not a spectacular endpoint.
# Also watch distill_diag: if distill_loss_last collapses to near zero the
# teacher is adding nothing, and if it dwarfs task_loss_last then alpha is too
# high and the student is ignoring the labels.
#
# The teacher costs one extra forward pass per step, no backward, so about
# 35% more per epoch: ten epochs in roughly half an hour.
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
    --n_epochs 10 \
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
    --lsq_scale_lr 1e-5 \
    --lsq_init mse \
    --lsq_grad_scaling N \
    --joint_lsq_metaq Y \
    --distillation Y \
    --distill_alpha 0.5 \
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
