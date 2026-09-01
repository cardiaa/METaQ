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

# test_264: EfficientNet-B0, full PRESTO on the test_261 recipe. ONE flag differs
# from test_262: --lsq_per_channel Y -> N.
#
# WHY. test_261 came back better than its own stopping rule asked for. Dropping
# per-channel step sizes for per-tensor ones cost 0.098 accuracy points on the
# mean of the last five epochs and bought 1.03 points of packet, a rate of 10.5
# points per point, as cheap as T2 at its knee and five times cheaper than the
# eight-bit depthwise trade.
#
#   test_258  4 bit, per-channel     11.51%  12.39% of model  -1.081
#   test_260  8-bit dw, per-channel  12.18%  13.05%           -0.635
#   test_261  8-bit dw, PER-TENSOR   11.15%  12.03%           -0.733
#
# HALF THE GAIN WAS NOT THE METADATA. The per-channel scales are a hard 0.599
# per cent of FP32 and that was the predicted saving, but the packet fell by
# 1.03: metadata -0.597, values -0.345, mask -0.094. One shared grid per tensor
# gives the coder ONE symbol distribution instead of one per output channel, and
# the pooled stream is more concentrated. Worth stating in the paper: per-channel
# quantization is paid for twice, once in metadata and once in code length.
#
# IT ALSO PUTS THIS NETWORK BACK IN LINE WITH THE OTHERS. ResNet-18, ResNet-50,
# DeiT-Small and ViT-B/16 all run per-tensor; EfficientNet-B0 was the only one on
# per-channel, inherited from the era when its depthwise layers needed the extra
# scales. Those layers now carry eight bits, so the precision comes from the
# codebook and the two mechanisms are no longer both required.
#
# WHERE THIS SHOULD LAND. The control is 12.03 per cent of the model at -0.733.
# T2 at the exposure-matched knee buys about 1.3 points of packet for 0.12 of
# accuracy, which reaches 10.73 per cent at -0.85, and T3 adds a few tenths more
# at 2.61 points per point. NNCodec is at (-0.89 @ 11.25 per cent) and their own
# frontier interpolates to -1.11 at 10.73 per cent. So unlike test_262, which was
# written as an honest tie, this run is aimed at BEATING them on both axes:
# smaller than 11.25 and above -0.89.
#
# test_262 STILL EARNS ITS THREE HOURS as the per-channel arm of the same pair,
# which turns "per-tensor is better" from one control comparison into a
# two-by-two with the regularizer on.
#
# THE DOSE IS UNCHANGED. T2 2.1e-7, the exposure-matched equivalent of the
# test_231 knee on a thirty-epoch cosine; T3 6e-8 with the relative dual step.
# The control's sparsity moved only from 15.48 to 16.55 per cent between the two
# branches, so the knee should sit at the same dose.
#
# COST: about three hours and twenty minutes. The six-hour wall is slack.

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
