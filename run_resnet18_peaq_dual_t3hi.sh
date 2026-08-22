#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_ObCTDoNN
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# test_237: ResNet-18, spending the accuracy headroom. ONE flag differs from
# test_236: --entropy_coeff 3e-8 -> 1e-7.
#
# WHY. test_236 is the best row of the project and it left money on the table:
#
#   test_174   LSQ only,          20 ep   69.888  +0.154   9.34%   10.71x
#   test_196   METaQ, slow dual,  20 ep   69.840  +0.106   6.08%   16.45x
#   test_235   METaQ, fast dual,  20 ep   69.668  -0.066   5.74%   17.42x
#   test_236   METaQ, fast dual,  40 ep   70.080  +0.346   5.73%   17.45x
#   test_197   METaQ, slow dual,  40 ep   69.578  -0.156   5.21%   19.19x
#
# test_236 finished 0.346 points ABOVE the full-precision checkpoint. That is
# headroom the compression never spent. This run spends it: 3.3 times the
# entropy coefficient, everything else identical, and the question is how far
# down the packet goes before the accuracy reaches parity.
#
# WHY 1e-7 IS NOT A WILD JUMP. The only 40-epoch run at a higher coefficient is
# test_197, at 6e-8, and its dual was 7.8 times slower than this one: at a fixed
# absolute step of 3e-9 it crossed its own dual interval 0.098 times per epoch
# against 0.76 here. So its entropy term was under-delivering, and its 5.21% is
# a floor rather than a ceiling for what 1e-7 with a converged dual can do.
# Whether the accuracy holds is exactly what is unknown.
#
# WHAT WOULD MAKE THIS THE HEADLINE. Below 5.34% of packet at 69.76 or above
# puts METaQ strictly ahead of HEMP+LOBSTER \citep{tartaglione2021hemp} on both
# axes, with one method where they stack two. test_236 already clears their
# accuracy by 0.38 but is 0.18MB larger; this run is aimed at the size.
#
# WHAT WOULD MAKE IT A FRONTIER POINT INSTEAD, which is still worth having: the
# accuracy lands below FP32 while the packet drops well under 5%. Then the row
# joins the accuracy-size curve next to test_236 and test_197, and the lossless
# headline stays test_236. The DeiT frontier in the paper is four points; the
# ResNet-18 one currently has three.
#
# CHECKS. xi_pinned_frac on epoch two: the relative dual step rescales with the
# coefficient, so 1e-7 should behave exactly as 3e-8 did, near zero after the
# first entropy epoch. If it stays high, halve --dual_step to 6.5e-3.
# Accuracy shape: test_236 dipped to 67.2 around epoch 12 and then climbed
# monotonically to 70.2. A deeper dip is expected here; if it is not clearly
# climbing by epoch 20, the dose is past the knee, but let it finish, because
# the endpoint is the frontier point.
#
# COST: 432s/epoch measured, about five hours for forty epochs.

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
    --n_epochs 40 \
    --lr 1e-2 \
    --optimizer_weight_decay 1e-4 \
    --perspective_coeff 1e-5 \
    --entropy_coeff 1e-7 \
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
