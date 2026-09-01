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

# This script launches a 4-node / 16-GPU AlexNet pretrained compression test.
# It stores only rank-0 output in a numbered log file under $WORK/acardia0/LeonardoTests.

DELTA=$1
BATCH_SIZE=$2
T1=$3
T2=$4
EPOCH_FRACTION=$5
C=$6
PRETRAINED=$7
GAMMA=$8
# test_113: perspective reformulation. Optional positional args (defaults given).
T3=${9:-1e-4}
PERSPECTIVE=${10:-N}
MAG_RATIO=${11:-0.5}
TARGET_SPARSITY=${12:-0}
N_EPOCHS=${13:-20}
SPARSITY_WARMUP=${14:-0}
RAMP_POWER=${15:-1.0}
CONV_SPARSITY=${16:-}
FC_SPARSITY=${17:-}
LAYER_SPARSITY=${18:-}
# test_133: hold lr and T2 flat for the whole run (decouples cumulative exposure
# from its per-epoch rate).  Default N keeps every earlier test reproducible.
FLAT_SCHEDULE=${19:-N}
# test_134: ascent step for the entropy dual, on the layer-size-normalized
# supergradient (replaces the old unscaled 1/subgradient_step = 1e-5).
DUAL_STEP=${20:-0.5}
# test_135: apply phi as a proximal operator on the weights (proximal gradient)
# instead of summing beta* into the loss gradient.  PROX_GAMMA is its step.
PROX=${21:-N}
PROX_GAMMA=${22:-1e-7}
# test_139: first epoch (0-based) at which the PROXIMAL step runs.  Kept separate
# from entropy_warmup_epochs, which also gates the grid reset and the start of the
# sparsity ramp (delaying that would delay pruning too).  Setting it late lets the
# ramp and the stabilisation run exactly as in the T2=0 baseline, then turns the
# prox on only to compress the values; it also cuts cost sharply, since a prox
# epoch costs ~5.6x a plain one (884s vs 157s).
PROX_START=${23:-0}
# test_143: iterative prune-and-heal schedule (stages ';'-separated, each
# "reach:hold:s1,...,s8", 1-based epochs).  Replaces the smooth ramp when set.
SPARSITY_SCHEDULE=${24:-}
# test_144: freeze the pruned index set per plateau (Deep-Compression style).
FREEZE_MASK=${25:-N}
# test_146: optimize the sparse subnetwork directly (pruned weights held at zero,
# zero gradient); only survivors are updated.  Main methodological gap vs DC.
TRAIN_SPARSE=${26:-N}
# Accuracy-preserving experiment: match Deep Compression's 8-bit convolutional
# and 5-bit fully-connected quantization, then let validation accuracy control
# how far along the requested sparsity vector the next epoch may move.
LAYER_C=${27:-}
ADIABATIC_TARGET=${28:-}
ADIABATIC_TOLERANCE=${29:-0.2}
ADIABATIC_STEP=${30:-0.02}
ADIABATIC_BACKOFF=${31:-0.04}
ADIABATIC_PATIENCE=${32:-2}
# test_150: Deep-Compression-style trained codebook. Cluster assignments are
# frozen after grid adaptation and only the shared centroid values are trained.
TRAIN_CENTROIDS=${33:-N}
# test_151: the true shared-centroid gradient is a bucket SUM and therefore needs
# its own learning-rate scale relative to ordinary per-weight fine-tuning.
CENTROID_LR_SCALE=${34:-1.0}
# test_153: Lloyd refinement of the linear codebook before assignments freeze.
CENTROID_KMEANS_ITERATIONS=${35:-0}
# Two-stage QAT -> fixed-codebook conversion (1-based epoch, 0 = immediate).
CENTROID_FREEZE_EPOCH=${36:-0}

if [ -z "$DELTA" ] || [ -z "$BATCH_SIZE" ] || [ -z "$T1" ] || [ -z "$T2" ] || [ -z "$EPOCH_FRACTION" ] || [ -z "$C" ] || [ -z "$PRETRAINED" ] || [ -z "$GAMMA" ]; then
    echo "Usage: sbatch run_alexnet_test_pretrained.sh <delta> <batch_size> <T1> <T2> <epoch_fraction> <C> <pretrained> <gamma> [T3] [perspective Y/N] [mag_prune_ratio] [target_sparsity] [n_epochs] [sparsity_warmup_epochs] [sparsity_ramp_power] [conv_sparsity] [fc_sparsity] [layer_sparsity csv] [flat_schedule Y/N] [dual_step] [prox Y/N] [prox_gamma] [prox_start] [sparsity_schedule] [freeze_mask Y/N] [train_sparse Y/N] [layer_C csv] [adiabatic_target] [adiabatic_tolerance] [adiabatic_step] [adiabatic_backoff] [adiabatic_patience] [train_centroids Y/N] [centroid_lr_scale] [centroid_kmeans_iterations] [centroid_freeze_epoch]"
    echo "Example (test_124 Deep-Compression per-layer): sbatch run_alexnet_test_pretrained.sh -10 64 1e-3 0 1.0 16 Y 2 1e-4 Y 0.5 0 25 10 0.5 0 0 0.16,0.62,0.65,0.63,0.63,0.91,0.91,0.75"
    exit 1
fi

# Optional per-layer sparsity flags (only passed if provided).
PERLAYER_FLAGS=""
if [ -n "$CONV_SPARSITY" ]; then PERLAYER_FLAGS="$PERLAYER_FLAGS --conv_sparsity $CONV_SPARSITY"; fi
if [ -n "$FC_SPARSITY" ]; then PERLAYER_FLAGS="$PERLAYER_FLAGS --fc_sparsity $FC_SPARSITY"; fi
if [ -n "$LAYER_SPARSITY" ]; then PERLAYER_FLAGS="$PERLAYER_FLAGS --layer_sparsity $LAYER_SPARSITY"; fi
if [ -n "$LAYER_C" ]; then PERLAYER_FLAGS="$PERLAYER_FLAGS --layer_C $LAYER_C"; fi
if [ -n "$ADIABATIC_TARGET" ]; then
    PERLAYER_FLAGS="$PERLAYER_FLAGS --adiabatic_accuracy_target $ADIABATIC_TARGET --adiabatic_accuracy_tolerance $ADIABATIC_TOLERANCE --adiabatic_step $ADIABATIC_STEP --adiabatic_backoff $ADIABATIC_BACKOFF --adiabatic_patience $ADIABATIC_PATIENCE"
fi
# NB: the schedule contains ';' and must NOT go through PERLAYER_FLAGS -- there it
# is injected unquoted and the inner `bash -lc` reads ';' as a command separator
# (test_143 first launch died with "30:40:...: command not found").  It is instead
# passed as its own torchrun flag, wrapped in double quotes that survive into the
# inner shell (see the --sparsity_schedule line below).

LOG_DIR=$WORK/acardia0/LeonardoTests
mkdir -p "$LOG_DIR"

LAST=$(find "$LOG_DIR" -maxdepth 1 -type f -name 'Leonardo_test_*.log' -printf '%f\n' | grep -oE '^Leonardo_test_[0-9]+\.log$' | grep -oE '[0-9]+' | sort -n | tail -1)

if [ -z "$LAST" ]; then
    NEXT=1
else 
    NEXT=$((LAST+1))
fi

LOG_FILE=$LOG_DIR/Leonardo_test_${NEXT}.log

module load profile/deeplrn
module load cineca-ai/4.3.0

export OMP_NUM_THREADS=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export PYTHONWARNINGS="ignore::UserWarning"
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

cd "$WORK/acardia0/METaQ"

srun --ntasks=$SLURM_NTASKS --ntasks-per-node=1 bash -lc '
    if [ "$SLURM_NODEID" -eq 0 ]; then
        torchrun \
            --nnodes=$SLURM_JOB_NUM_NODES \
            --nproc_per_node=4 \
            --node_rank=$SLURM_NODEID \
            --rdzv_backend=c10d \
            --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
            --rdzv_id=$SLURM_JOB_ID \
            train_on_gpus_pretrained.py \
            --model_name AlexNet \
            --delta '"$DELTA"' \
            --gamma '"$GAMMA"' \
            --data_root /leonardo_work/IscrC_ObCTDoNN/acardia0/datasets \
            --train_workers 4 \
            --val_workers 2 \
            --batch_size '"$BATCH_SIZE"' \
            --T1 '"$T1"' \
            --T2 '"$T2"' \
            --epoch_fraction '"$EPOCH_FRACTION"' \
            --n_epochs '"$N_EPOCHS"' \
            --lr 1e-4 \
            --max_iterations 3 \
            --metrics_interval 1 \
            --entropy_warmup_epochs 1 \
            --prox_start_epoch '"$PROX_START"' \
            --entropy_every 4 \
            --check_ddp_sync \
	    --C '"$C"' \
            --pretrained '"$PRETRAINED"' \
            --T3 '"$T3"' \
            --perspective '"$PERSPECTIVE"' \
            --mag_prune_ratio '"$MAG_RATIO"' \
            --target_sparsity '"$TARGET_SPARSITY"' \
            --sparsity_warmup_epochs '"$SPARSITY_WARMUP"' \
            --sparsity_ramp_power '"$RAMP_POWER"' \
            --flat_schedule '"$FLAT_SCHEDULE"' \
            --dual_step '"$DUAL_STEP"' \
            --prox '"$PROX"' \
            --prox_gamma '"$PROX_GAMMA"' \
            --sparsity_schedule "'"$SPARSITY_SCHEDULE"'" \
            --freeze_mask '"$FREEZE_MASK"' \
            --train_sparse '"$TRAIN_SPARSE"' \
            --train_centroids '"$TRAIN_CENTROIDS"' \
            --centroid_lr_scale '"$CENTROID_LR_SCALE"' \
            --centroid_kmeans_iterations '"$CENTROID_KMEANS_ITERATIONS"' \
            --centroid_freeze_epoch '"$CENTROID_FREEZE_EPOCH"' \
            '"$PERLAYER_FLAGS"' \
            > '"$LOG_FILE"' 2>&1
    else
        torchrun \
            --nnodes=$SLURM_JOB_NUM_NODES \
            --nproc_per_node=4 \
            --node_rank=$SLURM_NODEID \
            --rdzv_backend=c10d \
            --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
            --rdzv_id=$SLURM_JOB_ID \
            train_on_gpus_pretrained.py \
            --model_name AlexNet \
            --delta '"$DELTA"' \
            --gamma '"$GAMMA"' \
            --data_root /leonardo_work/IscrC_ObCTDoNN/acardia0/datasets \
            --train_workers 4 \
            --val_workers 2 \
            --batch_size '"$BATCH_SIZE"' \
            --T1 '"$T1"' \
            --T2 '"$T2"' \
            --epoch_fraction '"$EPOCH_FRACTION"' \
            --n_epochs '"$N_EPOCHS"' \
            --lr 1e-4 \
            --max_iterations 3 \
            --metrics_interval 1 \
            --entropy_warmup_epochs 1 \
            --prox_start_epoch '"$PROX_START"' \
            --entropy_every 4 \
            --check_ddp_sync \
	    --C '"$C"' \
            --pretrained '"$PRETRAINED"' \
            --T3 '"$T3"' \
            --perspective '"$PERSPECTIVE"' \
            --mag_prune_ratio '"$MAG_RATIO"' \
            --target_sparsity '"$TARGET_SPARSITY"' \
            --sparsity_warmup_epochs '"$SPARSITY_WARMUP"' \
            --sparsity_ramp_power '"$RAMP_POWER"' \
            --flat_schedule '"$FLAT_SCHEDULE"' \
            --dual_step '"$DUAL_STEP"' \
            --prox '"$PROX"' \
            --prox_gamma '"$PROX_GAMMA"' \
            --sparsity_schedule "'"$SPARSITY_SCHEDULE"'" \
            --freeze_mask '"$FREEZE_MASK"' \
            --train_sparse '"$TRAIN_SPARSE"' \
            --train_centroids '"$TRAIN_CENTROIDS"' \
            --centroid_lr_scale '"$CENTROID_LR_SCALE"' \
            --centroid_kmeans_iterations '"$CENTROID_KMEANS_ITERATIONS"' \
            --centroid_freeze_epoch '"$CENTROID_FREEZE_EPOCH"' \
            '"$PERLAYER_FLAGS"' \
            > /dev/null 2>&1
    fi
'
