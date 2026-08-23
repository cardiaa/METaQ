#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_ObCTDoNN
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=09:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# test_242: ResNet-50, first METaQ run. The ResNet-18 recipe transferred with the
# model name changed and the budget cut to twenty epochs for a first pass.
#
# WHY THIS CELL. ResNet-50 is the only network where two published methods report
# size after lossless coding, our own metric, and both pay for it:
#
#   DeepCABAC \citep{wiedemann2019deepcabac}   74.51   10.14%   -1.62 from 76.13
#   HEMP \citep{tartaglione2021hemp}           74.52    8.88%   -1.61
#   Deep Compression \citep{han2016deep}       68.95    6.15%   -7.18
#
# On ResNet-18 METaQ halves the packet of its own LSQ control and stays above the
# full-precision checkpoint. If that transfers even partially, this row beats both
# on both axes at once, which no other cell of the paper can claim.
#
# WHAT TRANSFERS AND WHAT DOES NOT. The learning rate, the schedule and the batch
# come from ResNet-18 unchanged: same family, same torchvision recipe, and the
# from-scratch rate divided by forty gives the same 1e-2 at global batch 1024.
# The launcher's own defaults for this model, lr 1e-5 and batch 32, are the
# pre-diagnosis values and are overridden here.
#
# --dual_step 1.3e-2 relative transfers exactly: in relative mode the dual
# convergence rate is dual_step * (1/C) * N_dual * calls_per_epoch, and ResNet-50
# has the same 1251 steps and entropy_every 4, hence the same 313 calls, as
# ResNet-18.
#
# T3 = 6e-8 IS A GUESS, and it is the one thing in this script that may be wrong.
# It is ResNet-18's headline dose, carried over because the two networks share a
# family and a weight distribution, but absolute coefficients have not
# transferred between distant architectures before. Twenty epochs make this a
# scouting run: if it lands past the knee, the dose comes down; if it lands short,
# it goes up, and the frontier is traced from there.
#
# COST: estimated near 1000s per epoch. ResNet-50 carries 2.18x the quantized
# weights of ResNet-18 across 2.57x the tensors, and its forward pass is about
# 2.3x heavier, so both the training and the dual solver scale. Roughly five and
# a half hours for twenty epochs; the nine-hour wall is slack on an estimate that
# has never been measured on this model.

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
