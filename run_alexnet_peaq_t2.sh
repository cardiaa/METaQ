#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_ObCTDoNN
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# test_229: AlexNet, second rung of the ladder. LSQ baseline (test_225) plus the
# sparsity term T2. The entropy term T3 stays off.
#
# Baseline to beat: test_225, A_Q 56.552 against 56.524 FP32, packet 10.75%
# (9.30x), 16.11% sparsity emerging from the LSQ zero bin alone.
#
# DOSE. One knob, --t2_scale. The per-layer targets set the shape of the push
# across layers; t2_scale sets its magnitude, and its value is read off the
# frontier that test_228 measured one epoch at a time as its ramp climbed:
#
#   t2_scale   sparsity   packet   accuracy lost (mid-ramp)
#     0.08       28%       8.3%      -0.05
#     0.16       40%       6.6%      -0.12
#     0.23       50%       5.5%      -0.47
#     0.31       58%       4.7%      -0.85   <-- this run
#     0.39       65%       4.1%      -1.36
#     0.47       71%       3.6%      -1.64
#
# The budget for repaying that loss is what annealing is worth on this recipe:
# +0.76, measured on test_225 itself, 55.790 at epoch 13 to 56.552 at epoch 60.
# So the lossless zone sits around 50-60% sparsity, not the 89% of Deep
# Compression, which buys its extra room with a different representation: 8-bit
# convolutions, 5-bit fully connected, k-means weight sharing and iterative
# prune-retrain cycles. More bits per surviving weight, so more weights can go.
#
# The frontier rows are pessimistic, though: each was measured mid-ramp on a
# network driven from 15.7% to 71% sparsity in six epochs with no settling phase
# at all. This run ramps over forty epochs and then holds the dose for twenty
# more with the rate on its floor, so the same sparsity should cost less.
#
# SCHEDULE. T2 live from epoch one, since test_225 already served as the
# diagnostic phase and repeating it would waste ten epochs; forty-epoch ramp;
# twenty epochs at fixed T2 and at min_lr for the network to settle into the
# compressed configuration.
#
# Everything else is the frozen test_225 recipe, plus --min_lr 1e-4, which the
# phase schedule requires: at the historical 1% floor the flat phase would run at
# 2e-5, where test_225 measured a level-change rate of 0.115% per epoch, i.e. a
# frozen network being asked to absorb full-dose compression.
#
# IF IT LANDS LOSSLESS, raise t2_scale one row. IF IT LANDS SHORT BY A FEW
# TENTHS, drop to 0.23. Either way the next dose is one line.

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
        --model_name AlexNet \
        --delta -10 \
        --gamma 2 \
        --data_root /leonardo_work/IscrC_ObCTDoNN/acardia0/datasets \
        --train_workers 4 \
        --val_workers 2 \
        --batch_size 64 \
        --n_epochs 60 \
        --diagnostic_epochs 0 \
        --metaq_ramp_epochs 40 \
        --metaq_flat_epochs 20 \
        --lr 2e-3 \
        --lr_warmup_epochs 1 \
        --min_lr 1e-4 \
        --optimizer_weight_decay 1e-4 \
        --perspective_coeff 1e-5 \
        --entropy_coeff 0 \
        --sparsity_coeff 0 \
        --layerwise_t2_targets 0.112,0.434,0.455,0.441,0.441,0.637,0.637,0.525 \
        --t2_calibration displacement \
        --t2_scale 0.31 \
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
        --bn_recalibration_batches 0 \
        --C 16 \
        --layer_C 256,256,16,16,16,16,16,16 \
        --max_iterations 3 \
        --metrics_interval 1 \
        --entropy_warmup_epochs 0 \
        --entropy_every 4 \
        --dual_step 3e-9 \
        --check_ddp_sync \
        --pretrained Y \
        > "$OUTPUT_TARGET" 2>&1
'
