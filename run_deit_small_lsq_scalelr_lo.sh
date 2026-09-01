#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_ObCTDoNN
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=0:30:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# test_181: are frozen step sizes better still, or have we bracketed the optimum?
#
# The step size a of a tensor fixes where its sixteen representable levels sit:
# -8a, -7a, ..., -a, 0, a, ..., 7a. LSQ's whole point is that a is LEARNED
# rather than fixed, so it needs its own learning rate, and lsq_scale_lr is
# exactly that: how fast the quantization levels are allowed to move. Too slow
# and the levels stay where initialization put them; too fast and the grid
# chases the weight distribution, shifting under the network before the weights
# can settle.
#
# The 1e-5 we had used since ResNet-18 turns out not to be the right value for a
# transformer. Comparing the average of the first four epochs with that of the
# last three, over ten epochs, plain LSQ trends by -0.07 at 1e-5 (test_173),
# +0.11 at 1e-4 (test_180) and +0.22 at 1e-6 (test_179), the last reaching
# 79.208 and still climbing. That is the first rising trend we have seen on
# DeiT, after granularity, initialization and bit allocation each failed to move
# the plateau. Measured displacement of the scales from initialization to epoch
# ten: 3.8% at 1e-6 against 10.5% at 1e-4.
#
# This run goes one decade further down, to 1e-7, where the scales are
# essentially pinned to their MSE initialization. It brackets the optimum, and
# the two outcomes are both informative. If accuracy climbs as it does at 1e-6,
# then learning the step sizes contributes nothing on this architecture and they
# may as well be frozen, which is a blunt but publishable statement about LSQ on
# vision transformers. If it climbs less, the optimum sits near 1e-6 and the
# long run should use that.
#
# Everything else is test_173: plain LSQ, per-tensor, MSE initialization, four
# bits on all fifty tensors, no regularizer.
#
# WHAT TO READ, against the first ten epochs of test_173, which run
# 79.056, 79.120, 79.044, 79.110, 78.996, 78.984, 78.986, 78.998, 78.976, 79.068
# and therefore peak by epoch 4 and settle at 78.99: the SIGN and size of the
# trend, not the endpoint, since the endpoint spread across these runs is within
# the run-to-run noise we have observed.
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
    --lsq_scale_lr 1e-7 \
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
