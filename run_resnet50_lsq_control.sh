#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_ObCTDoNN
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=05:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# test_242: ResNet-50, LSQ-only at twenty epochs. First run on this network,
# and it goes alone: the PRESTO run follows once this one has landed and has
# told us what the learning rate does here.
#
# THIS RUN DOES TWO JOBS AT ONCE, which is why it is worth its own slot rather
# than being skipped.
#
# It is the denominator the cell needs. A ResNet-50 row means nothing on its own:
# the claim the paper makes everywhere else is against a matched control at the
# same budget and the same byte accounting, and that is what makes DeepCABAC and
# HEMP comparable to us in the first place.
#
# And it verifies the learning rate before the expensive run spends it. The rate
# here is transferred from ResNet-18 by family, not measured on this model, and
# the launcher's own default for ResNet-50 is lr 1e-5 at batch 32, which is the
# same class of error that cost weeks on AlexNet and EfficientNet-B0. On both of
# those the symptom was visible in the control alone: the latent weights did not
# move, the first percentile shifted by hundredths of a percent, and A_Q was PTQ
# plus noise. Watch weight_percentiles here: a few percent of movement over the
# run means the rate is right, near zero means it is not.
#
# WHAT TO EXPECT. On ResNet-18 four-bit LSQ finishes ABOVE the full-precision
# checkpoint, by 0.154 at twenty epochs. If ResNet-50 does the same there is
# headroom to spend on compression, and the published rows above, which all sit
# 1.6 points below their baseline, are beatable on both axes.
#
# COST: never measured on this model. ResNet-50 carries 2.18x the quantized
# weights of ResNet-18 across 2.57x the tensors and its forward pass is about
# 2.3x heavier, so the estimate is near 320s per epoch, under two hours for
# twenty. The five-hour wall is slack on an estimate, not a prediction: the
# number this run produces is what sizes the PRESTO run that follows, which
# adds the dual solver on top.

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
    --perspective_coeff 0 \
    --entropy_coeff 0 \
    --sparsity_coeff 0 \
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
