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

# test_184: how far does the distillation dose go?
#
# Test_183 removed the sag that every DeiT run had shown. Where plain LSQ
# (test_173) drops to 78.98 across epochs five to nine, distillation climbs to
# 79.42 and peaks at 79.542, lifting the level over the last three epochs from
# 79.014 to 79.473. That is +0.46 against an FP32 baseline of 79.742, so a gap
# of -0.27, matching the -0.3 that Q-ViT reports for LSQ on DeiT-S, reached in
# ten epochs rather than three hundred. The limit was never the quantizer, whose
# four knobs we swept without moving the plateau, nor the budget, which made
# things worse: it was that nothing anchored the network to the function we are
# trying to preserve.
#
# The dose is still gentle. In test_183 distill_loss_last sits between 0.056 and
# 0.088 while task_loss_last sits between 0.47 and 0.85, so the distillation
# term is about ten times smaller in magnitude. At alpha = 0.5 the two are
# weighted equally in the convex combination, which means the anchor still
# contributes roughly a tenth of the gradient. Raising alpha to 0.9 makes their
# effective contributions comparable, and since half strength already bought
# 0.46 it is worth measuring where the curve turns.
#
# Only alpha changes. The regularizer stays off, so this remains a question
# about the training signal alone.
#
# WHAT TO READ, against the ten epochs of test_183, which run 79.114, 79.360,
# 79.192, 79.380, 79.416, 79.424, 79.460, 79.462, 79.542, 79.414: the level over
# the last three epochs, 79.473. Above it, the dose was not yet saturated and
# the extra headroom can later be spent on compression. Below it, alpha = 0.5
# was already past the optimum and the student is starting to ignore the labels,
# which distill_diag will show as distill_loss_last dominating task_loss_last.
#
# The teacher costs no measurable time: test_183 ran at 134s per epoch against
# 143s for test_173. Ten epochs in about half an hour.
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
    --lsq_init mse \
    --lsq_grad_scaling N \
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
