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

# test_260: EfficientNet-B0, LSQ-only control with the sixteen depthwise
# convolutions and the input convolution at eight bits. ONE flag differs from
# test_258: --layer_C.
#
# WHY. test_258 killed the hypothesis this cell was built on. Tripling the
# budget from ten epochs to thirty bought NOTHING:
#   test_230  10 ep, 1 per cent lr floor   76.590   -0.976   11.55%
#   test_258  30 ep, 5 per cent lr floor   76.487   -1.079   11.51%
# and the second run is if anything the worse of the two. It is not that the
# network failed to train: the first percentile of its latent weights moved
# -8.93 per cent against test_230's -3.07, so it trained THREE TIMES harder and
# landed in the same place. level_change held near 4.5 per cent for twenty
# epochs before the cosine put it away. This is a representational ceiling, not
# an optimization one, and no amount of budget will move it.
#
# WHERE THE CEILING IS, measured. Clipping fractions at epoch 30, the sixteen
# depthwise tensors against the other sixty-six:
#   depthwise   mean 6.58%   max 28.91%     (epoch 1: mean 5.50%)
#   everything else   mean 1.93%   max 9.14%   (epoch 1: mean 2.43%)
# Three and a half times the clipping, and moving the WRONG WAY over training
# while the rest of the network improves. Five of the eight worst layers in the
# network are depthwise, and features.7.0.block.1.0, the 5x5 depthwise of the
# last MBConv stage, ends with 28.91 per cent of its weights pinned to the grid
# endpoints. A depthwise kernel gives each output channel a single input
# channel, so its quantization error never averages over a sum; this is the
# textbook failure mode and here it is in the log.
#
# WHAT IT COSTS. The sixteen depthwise tensors are 182,016 parameters, 3.48 per
# cent of the quantized weights, so four extra bits each is at most 0.4345 per
# cent of FP32 added to the packet, and less than that after entropy coding. The
# input convolution is 864 parameters and free. The classifier, which is 24.45
# per cent of the network on its own, stays at four bits: taking it to eight
# would cost 3.06 per cent of FP32 and end the compression claim.
#
# WHAT WE NEED FROM IT. NNCodec delivers -0.89 at 11.25 per cent of the whole
# model. We are at -1.08 and 12.39 per cent. T2 at its measured knee buys about
# 1.3 points of packet for 0.12 of accuracy, which reaches 11.1 per cent but at
# -1.20, still behind them on accuracy. The packet is affordable; the accuracy
# is what is missing. This run has to return at least 0.4 points for the cell to
# be winnable. Anything under 0.2 and the honest move is to stop chasing
# EfficientNet-B0 and report it as the architecture where the method does not
# reach the frontier, with the clipping table above as the reason.
#
# ATTRIBUTION IS DELIBERATELY BUNDLED. Seventeen tensors move together, and the
# depthwise carry 99.5 per cent of the cost, so this is a yes/no probe on the
# ceiling rather than an attribution study. If it works, the input convolution
# can be split out afterwards for a few minutes of compute.
#
# COST: 158s per epoch measured on test_258, so about eighty minutes. Eight-bit
# tensors do not change the forward cost. The three-hour wall is slack.

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
