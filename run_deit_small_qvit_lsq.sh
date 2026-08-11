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

# test_178: does the Q-ViT LSQ CONFIGURATION change the sign of the trend?
#
# Reading their released code (github.com/YanjingLi0202/Q-ViT: Quant.py,
# _quan_base.py, quant_vision_transformer.py) removes the guesswork about what
# their LSQ actually is. Four things differ from ours:
#
#   1. PER-CHANNEL weights. Every block tensor is built with
#      mode=Qmodes.kernel_wise, and _LinearQ then allocates
#      alpha = Parameter(torch.Tensor(out_features)).
#   2. Canonical init alpha = 2*mean(|w|)/sqrt(Qp) rather than an MSE grid search.
#   3. Gradient scale g = 1/sqrt(numel*Qp) applied to the step size through a
#      straight-through trick, with alpha a plain Parameter and therefore inside
#      the MAIN optimizer at the network learning rate.
#   4. Eight bits on the patch embedding and the head (head is LinearQ(...,
#      nbits_w=8)); we quantize both at four.
#
# Items 1, 2 and 4 are flags we already have, so this run covers three of the
# four without touching any code. Item 3 is structural, since our step sizes
# live in a separate Adam at 1e-5 where a constant gradient factor is a no-op,
# and it is deliberately left for a follow-up.
#
# The regularizer stays off: this asks a question about the quantizer, and the
# entropy solver would triple the cost of asking it.
#
# WHAT TO READ. Not the level, the SIGN. Q-ViT reaches 79.6 with 300 epochs,
# LAMB and the DeiT augmentation pipeline, none of which are here. Test_173 and
# test_177 both peak by epoch 5 and then decline monotonically, test_177 losing
# 1.13 points over fifty epochs. If accuracy instead RISES between epoch 1 and
# epoch 10, the quantizer configuration was the blocker and the same change
# should be carried to ResNet-18. If it peaks and declines again, the remaining
# suspects are the step-size optimizer and the missing augmentation.
#
# Known minor deviations: their init copies one global mean into every channel,
# ours fits each channel separately; and their 8-bit patch embedding and head
# stay layer-wise while --lsq_per_channel makes every tensor per-channel.
#
# Ten epochs at about 143s each: roughly 25 minutes.
#
# The configuration matches test_171 exactly except that all three coefficients
# are zero. That combination is what actually disables the regularizer: the
# closed-form branch guarding the perspective ridge and the sparsity term
# requires one of them to be nonzero, and the entropy branch requires a positive
# entropy coefficient, so neither contributes a gradient. Note that
# entropy_warmup_epochs alone would NOT have been enough, since it postpones the
# entropy coefficient only and leaves the other two terms active, which is why
# the first epoch of tests 169 to 171 was not a plain-LSQ measurement. The
# perspective flag itself stays Y because joint LSQ-METaQ requires it and
# because turning it off would enable the legacy non-perspective dual instead.
#
# Reading against the 78.844 that test_171 reached at epoch 18: a plain-LSQ
# result near 79.7 means the missing accuracy is the METaQ dose and the fix is
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
    --lsq_init lsq \
    --lsq_grad_scaling N \
    --lsq_per_channel Y \
    --joint_lsq_metaq Y \
    --bn_recalibration_batches 0 \
    --C 16 \
    --layer_C 256,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,256 \
    --max_iterations 3 \
    --metrics_interval 1 \
    --entropy_warmup_epochs 1 \
    --entropy_every 4 \
    --dual_step 3e-9 \
    --check_ddp_sync \
    --pretrained Y \
    > "$OUTPUT_TARGET" 2>&1
'
