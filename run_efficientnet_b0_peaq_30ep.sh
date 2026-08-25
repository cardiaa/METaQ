#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_ObCTDoNN
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=06:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# test_259: EfficientNet-B0, full METaQ at thirty epochs, against the matched
# control test_258. This run has one target and it is a published number.
#
# THE TARGET. NNCodec, the reference implementation of ISO/IEC 15938-17, codes
# EfficientNet-B0 to 2.41MB of 21.45MB at 0.89 points below its checkpoint.
# In the accounting of this paper that is (-0.89 @ 11.24 per cent of the whole
# model), or 8.90x. Our ten-epoch LSQ control is (-0.976 @ 12.43 per cent), so
# we are currently behind them on BOTH axes on this network, and this is the
# only architecture in the paper where that is true.
#
# WHY WE SHOULD PASS THEM. Three separate measurements say the gap is budget
# and dose, not mechanism.
#   1. Ten epochs against the twenty to hundred everywhere else, with the
#      accuracy curve still rising when the schedule ended.
#   2. test_231 traced the T2 frontier: 22.96 per cent sparsity costs 0.12
#      points against the matched baseline and takes the packet from 11.55 to
#      10.26 per cent; 25.15 per cent costs 0.42 and reaches 9.89.
#   3. test_234 proved the entropy channel opens here once the dual step is
#      relative: values_zstd_ratio fell 0.649 against the control's 0.099 over
#      the same ten epochs. The earlier verdict that T3 was inert on this
#      network was an artefact of the absolute dual step, not a property of it.
#
# THE T2 DOSE IS EXPOSURE-MATCHED, NOT COPIED. A ramp row reports the exposure
# accumulated up to that epoch, not a coefficient held for a whole run: the
# test_231 knee reads 1e-6 at epoch five but reproduces on a twenty-epoch
# cosine only at 3.16e-7. The schedule sums with lr_warmup 2 and a five per
# cent floor are 10.9 over twenty epochs and 16.2 over thirty, so the same
# exposure on this schedule is 2.1e-7, which is what is set below. The
# arithmetic is in scratchpad/verify_t2_exposure.py; the lesson behind it is
# that on this project the dose is measured and then transported by its
# integral, never read off another run's coefficient.
#
# T3 = 6e-8, twice the test_234 dose, with the relative dual step that made it
# work. The ResNet-18 ablation says the two terms barely add, 4.80 per cent
# together against 4.94 for T3 alone, so T3 is expected to contribute a few
# tenths of a point of packet here rather than to carry the run. T2 is the
# strong channel on this network, as on AlexNet and unlike on both ResNets.
#
# WHAT WOULD COUNT AS SUCCESS: anything at or under 11.2 per cent of the whole
# model at better than -0.89. The projection from the two frontiers above is
# 10.5 to 10.8 per cent at -0.65 to -0.75, which would take the last cell where
# a published method is ahead of us.
#
# WATCH THE METADATA. The per-channel step sizes over 82 tensors are a hard
# floor of 0.599 per cent of the packet, identical in test_230 and test_231.
# That was five per cent of the packet at the baseline and becomes six per cent
# at the target, and it is a cost NNCodec does not pay because its local
# scaling factors are merged into the folded BatchNorm parameters. We pay it
# and report it.
#
# COST: 388-397s per epoch measured on test_234, so about three hours and
# twenty minutes for thirty epochs. The six-hour wall is slack.
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
    if [ "$SLURM_NODEID" -eq 0 ]; then OUTPUT_TARGET="$LOG_FILE"; else OUTPUT_TARGET=/dev/null; fi
    "$METAQ_PYTHON" -m torch.distributed.run \
        --nnodes=$SLURM_JOB_NUM_NODES \
        --nproc_per_node=4 \
        --node_rank=$SLURM_NODEID \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        --rdzv_id=$SLURM_JOB_ID \
        train_on_gpus_pretrained.py \
        --model_name EfficientNet-B0 \
        --delta -10 \
        --gamma 2 \
        --data_root /leonardo_work/IscrC_ObCTDoNN/acardia0/datasets \
        --train_workers 4 \
        --val_workers 2 \
        --batch_size 256 \
        --n_epochs 30 \
        --lr 2e-2 \
        --lr_warmup_epochs 2 \
        --min_lr 1e-3 \
        --optimizer_weight_decay 1e-4 \
        --perspective_coeff 1e-5 \
        --entropy_coeff 6e-8 \
        --sparsity_coeff 2.1e-7 \
        --perspective Y \
        --flat_schedule N \
        --mag_prune_ratio 0 \
        --quantization Y \
        --quantizer lsq \
        --lsq_scale_lr 1e-3 \
        --lsq_scale_lr_mode relative \
        --lsq_scale_lr_schedule Y \
        --lsq_init mse \
        --lsq_grad_scaling N \
        --lsq_per_channel Y \
        --joint_lsq_metaq Y \
        --distillation Y \
        --distill_alpha 0.5 \
        --distill_tau 1.0 \
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
