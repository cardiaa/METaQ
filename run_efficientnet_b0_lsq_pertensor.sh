#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_ObCTDoNN
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=03:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# test_261: EfficientNet-B0, the test_260 recipe with PER-TENSOR step sizes
# instead of per-channel. ONE flag differs: --lsq_per_channel Y -> N.
#
# WHY. test_260 did what it was designed to do and it was not enough. Putting the
# sixteen depthwise convolutions at eight bits took their mean clipping from 6.58
# to 1.57 per cent, indistinguishable from the 1.84 of the rest of the network,
# and bought 0.362 points of accuracy on the final epoch, 0.446 on the mean of
# the last five. The diagnosis was right. But it cost 0.67 points of packet,
# almost all of it in the values stream, 8.89 to 9.47 per cent, because eight-bit
# tensors hand the coder a much wider alphabet.
#
# THE EXCHANGE RATES, all measured on this network, in points of packet bought
# per point of accuracy spent:
#   T2 up to its knee (test_231, 11.55 -> 10.26)      10.75
#   T3 (test_234, ten epochs at 3e-8)                  2.61
#   eight-bit depthwise, read in reverse               1.85
#   T2 beyond the knee (test_231 rows 3 to 6)          1.17
# So the eight-bit depthwise is worth having, but only after the two cheap
# channels are exhausted. Projected to NNCodec's operating point of 11.25 per
# cent of the model: the four-bit recipe lands at -1.19, the eight-bit-depthwise
# recipe at -1.03, and they are at -0.89. Better, and still short.
#
# WHAT IS LEFT IS THE METADATA, AND IT IS EXACTLY THE SIZE OF THE GAP. The
# per-channel step sizes over 82 tensors cost a hard 0.599 per cent of FP32,
# identical in test_230, test_231, test_258 and test_260 because it is a fixed
# count of fp32 numbers. That is five per cent of the packet at the baseline and
# it never shrinks. Per-tensor scales replace it with 82 numbers instead of one
# per output channel, a metadata ratio of 0.0016 per cent, so the flag alone is
# worth 0.597 points of packet. Priced at T3's exchange rate that is 0.23 points
# of accuracy, and we are 0.14 short.
#
# WHY IT IS WORTH TRYING NOW AND WAS NOT BEFORE. Per-channel exists to give each
# output channel its own scale where a shared one fits badly, and on this network
# that meant the depthwise layers, where one output channel sees one input
# channel. Those layers now carry eight bits, so the precision they needed comes
# from the codebook rather than from the metadata. The two mechanisms are
# substitutes and we are currently paying for both.
#
# NNCodec does not pay this. Their local scaling factors are trained and then
# merged into the folded BatchNorm parameters, so they add no coded data at all.
# If per-tensor works, we stop paying it honestly rather than by omission.
#
# THE STOPPING RULE, fixed before the run. Per-tensor buys 0.597 points of packet.
# If it costs less than about 0.15 accuracy points the cell is winnable and the
# METaQ run goes on this recipe. If it costs more than 0.3, stop: EfficientNet-B0
# is at parity with NNCodec and the paper says so, with the clipping table and
# the exchange rates as the explanation. That is a publishable result and a more
# interesting one than a narrow win.
#
# COST: 158s per epoch on test_258 and test_260 alike, so about eighty minutes.
# The three-hour wall is slack.

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
        --layer_C 256,256,16,16,16,16,256,16,16,16,16,256,16,16,16,16,256,16,16,16,16,256,16,16,16,16,256,16,16,16,16,256,16,16,16,16,256,16,16,16,16,256,16,16,16,16,256,16,16,16,16,256,16,16,16,16,256,16,16,16,16,256,16,16,16,16,256,16,16,16,16,256,16,16,16,16,256,16,16,16,16,16 \
        --lr 2e-2 \
        --lr_warmup_epochs 2 \
        --min_lr 1e-3 \
        --optimizer_weight_decay 1e-4 \
        --perspective_coeff 0 \
        --entropy_coeff 0 \
        --sparsity_coeff 0 \
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
        --lsq_per_channel N \
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
