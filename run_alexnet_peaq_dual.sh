#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_ObCTDoNN
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=20:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# test_249: AlexNet, PRESTO complete with the entropy dual allowed to converge.
# The parked test_229bis script with one flag added: --dual_step_mode relative,
# and --dual_step rescaled from 3e-9 absolute to 1.3e-2 relative.
#
# WHY IT IS WORTH RUNNING AT ALL, now that AlexNet is a transfer network and not
# a competitive row. One claim in our own notes rests on a measurement taken with
# a starved dual: "on AlexNet T2 is more efficient than T3". It was drawn from
# test_220, which crossed its dual interval 0.0135 times per epoch against the
# 0.195 of the ResNet-18 run where the entropy term worked. Fourteen times slow.
# Before that sentence goes into the paper it has to be re-measured with a dual
# that converges, and AlexNet has 1251 steps and entropy_every 4, hence the same
# 313 calls as ResNet-18, so 1.3e-2 transfers exactly.
#
# SIXTY EPOCHS, because it is the only budget that can be compared to test_225,
# the frozen LSQ baseline at 56.552, and because twenty cannot be lossless on
# this recipe at any dose (test_223 at -0.568, test_224 at -0.072, test_225 at
# +0.028).
#
# THE T2 DOSE IS UNCHANGED at --t2_scale 0.050, and it stays correct: the
# calibration weights the schedule sums by the duty cycle of the PRESTO gradient,
# which depends on entropy_every and not on how the dual step is expressed.
#
# COST: this is the expensive one. About 2.1s per dual call measured on AlexNet,
# 313 calls per epoch, so roughly 810-860s per epoch and thirteen to fourteen
# hours over sixty. If the queue is contended this is the run to drop: AlexNet is
# an appendix network and the row it would produce is a footnote, not a headline.
#
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
        --lr 2e-3 \
        --lr_warmup_epochs 1 \
        --optimizer_weight_decay 1e-4 \
        --perspective_coeff 1e-5 \
        --entropy_coeff 3e-8 \
        --sparsity_coeff 0 \
        --layerwise_t2_targets 0.112,0.434,0.455,0.441,0.441,0.637,0.637,0.525 \
        --t2_calibration displacement \
        --t2_scale 0.050 \
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
