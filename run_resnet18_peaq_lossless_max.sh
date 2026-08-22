#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_ObCTDoNN
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# test_239: ResNet-18, the lossless row at the largest dose that clears the
# checkpoint. ONE flag differs from test_238: --entropy_coeff 1e-7 -> 6e-8.
# Sixty epochs, unchanged.
#
# THE FRONTIER, against 69.734 FP32:
#
#   test_174   LSQ only,  20 ep   69.888  +0.154   9.34%   10.71x
#   test_236   T3=3e-8,   40 ep   70.080  +0.346   5.73%   17.45x
#   test_237   T3=1e-7,   40 ep   69.406  -0.328   4.21%   23.75x
#   test_238   T3=1e-7,   60 ep   69.606  -0.128   4.14%   24.15x
#
# THE DOSE MODEL, anchored on those points and reproducing test_238 exactly:
#
#     T3      acc at 60 ep   packet   factor
#    5e-8        +0.223       5.02%   19.9x
#    6e-8        +0.131       4.78%   20.9x   <- this run
#    7e-8        +0.053       4.59%   21.8x
#    8e-8        -0.015       4.42%   22.6x
#    1e-7        -0.128       4.14%   24.1x   (measured)
#
# It has two ingredients. Accuracy against log-dose comes from the two
# forty-epoch points. The value of the extra twenty epochs is measured at 1e-7,
# where test_238 gained 0.200 accuracy points over test_237, and it is scaled
# down for weaker doses because that gain is largely recovery from damage a
# strong dose does: the end-of-run slopes give the ratio, +0.047 per epoch at
# 3e-8 against +0.069 at 1e-7.
#
# WHY NOT 8e-8, WHICH IS WHAT THE COMPRESSION WOULD PREFER. The model puts it at
# -0.015, that is on the wrong side of the checkpoint, and 7e-8 at +0.053. Those
# margins are smaller than the model's own error, and far smaller than the 0.338
# of run-to-run noise measured at epoch one of test_235 against test_236, where
# the entropy term is off and the two runs are functionally identical. Buying
# 3.7 percent more compression with a seven-hour run whose result cannot be
# called lossless either way is the wrong trade. 6e-8 is the largest dose whose
# predicted margin survives a plausible model error.
#
# EXPECTED: about 4.8 percent of packet, that is 20.9x, at roughly +0.13 above
# the checkpoint. That replaces test_236 as the lossless headline and improves
# it by a fifth.
#
# COST: 432s/epoch measured, about seven hours and a quarter.

LOG_DIR=$WORK/acardia0/LeonardoTests
mkdir -p "$LOG_DIR"

# Log number fixed by agreement so that this run and its control can be
# submitted together: the auto-numbering reads the directory at start-up, and
# two jobs dispatched in the same instant would both claim the same number and
# one would overwrite the other.
LOG_ID=239
LOG_FILE=$LOG_DIR/Leonardo_test_${LOG_ID}.log
if [ -e "$LOG_FILE" ]; then
    echo "refusing to overwrite $LOG_FILE" >&2
    exit 1
fi
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
    --model_name ResNet-18 \
    --delta -10 \
    --gamma 2 \
    --data_root /leonardo_work/IscrC_ObCTDoNN/acardia0/datasets \
    --train_workers 4 \
    --val_workers 2 \
    --batch_size 64 \
    --n_epochs 60 \
    --lr 1e-2 \
    --optimizer_weight_decay 1e-4 \
    --perspective_coeff 1e-5 \
    --entropy_coeff 6e-8 \
    --sparsity_coeff 1e-7 \
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
