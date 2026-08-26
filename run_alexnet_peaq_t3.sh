#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_ObCTDoNN
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=10:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# test_265: AlexNet, the ENTROPY arm of the ablation, mirroring test_229 exactly.
# Same schedule, same recipe, same ramp: the sparsity machinery is off and the
# entropy coefficient is ramped in its place.
#
# WHY IT EXISTS. Frangioni asked, on 26 August, how much the pruning term
# contributes on its own, and specifically whether putting everything on pruning
# and nothing on compression gets us to the same place. On ResNet-18 that
# ablation is complete and answers him: pruning alone reaches 14.33x, entropy
# alone 19.88x, the two together 20.45x, so the entropy term is the one doing the
# work and the two do not stack. On AlexNet only half the ablation exists.
# test_229 IS the "everything on pruning" arm: 70.60 per cent sparsity, 3.68 per
# cent of the model, -1.586. This run is the missing mirror.
#
# WHAT CHANGES FROM test_229, and it is only these four flags:
#   --layerwise_t2_targets, --t2_calibration, --t2_scale   removed
#   --entropy_coeff 0 -> 3e-7, ramped by --metaq_ramp_epochs 40
#   --dual_step 3e-9 -> 1.3e-2 with --dual_step_mode relative
# The last one is not cosmetic. test_232 and test_233 proved that an ABSOLUTE
# dual step cannot drive the entropy term, because the dual range it must cross
# is proportional to entropy_coeff while the step is not: raising T3 tenfold left
# the applied gradient identical to four significant figures. test_229 carries
# the old absolute 3e-9, which was harmless there because T3 was off, and would
# make this run inert if copied over.
#
# WHY A RAMP AND NOT A FIXED DOSE. test_229 is a ramp, so a ramp is what makes
# the two arms comparable row by row: at every epoch both have accumulated the
# same schedule and we read directly which term buys more packet per point of
# accuracy. A fixed dose would give one point against their seven.
#
# THE TOP OF THE RAMP IS DELIBERATELY OVERDOSED, which is how test_228 and
# test_231 were designed and the only method that has worked on this project for
# finding a dose. At batch 64 on sixteen GPUs this network has 1251 steps per
# epoch and entropy_every 4, hence the same 313 dual calls as ResNet-18, where
# T3=1e-7 held for sixty epochs reaches 4.14 per cent. A ramp topping out at
# 3e-7 sweeps from far below that to well past it.
#
# WHAT WE EXPECT, and it is worth writing down because it is the interesting
# case. On ResNet-18 and ResNet-50 the entropy arm dominates the sparsity arm.
# On AlexNet we expect the opposite, because after T2 the surviving weights
# already carry only 1.12 bits each and there is nothing left for T3 to remove.
# If this run confirms it, the sentence "AlexNet is the wrong battlefield" stops
# being an intuition about fully connected layers and becomes a measurement, and
# the architecture-dependence of the two channels becomes a finding rather than
# an excuse.
#
# COST: the frozen recipe runs at 157s per epoch and the historical AlexNet runs
# with the entropy channel on cost about 390s at entropy_every 4, so budget 400s
# and roughly six hours and forty minutes for sixty epochs. The ten-hour wall is
# slack.

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
        --entropy_coeff 3e-7 \
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
        --bn_recalibration_batches 0 \
        --C 16 \
        --layer_C 256,256,16,16,16,16,16,16 \
        --max_iterations 3 \
        --metrics_interval 1 \
        --entropy_warmup_epochs 0 \
        --entropy_every 4 \
        --dual_step 1.3e-2 \
        --dual_step_mode relative \
        --check_ddp_sync \
        --pretrained Y \
        > "$OUTPUT_TARGET" 2>&1
'
