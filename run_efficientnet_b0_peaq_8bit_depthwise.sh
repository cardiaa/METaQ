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

# test_262: EfficientNet-B0, full PRESTO at thirty epochs on the test_260 recipe.
# Two flags differ from the run this replaces: --layer_C is now set, putting the
# sixteen depthwise convolutions and the input convolution at eight bits.
#
# WHY THE RECIPE CHANGED. This run was written against test_258, the four-bit
# control. test_260 then showed that the eight-bit depthwise recipe is strictly
# better at matched size: projected to NNCodec's 11.25 per cent of the model, the
# four-bit branch lands at -1.19 and this one at -1.03. Running PRESTO on the
# weaker of the two branches would have wasted the three hours.
#
# THE TARGET, unchanged. NNCodec codes EfficientNet-B0 to 2.41MB of 21.45MB at
# 0.89 points below its checkpoint: (-0.89 @ 11.25 per cent of the whole model),
# 8.89x. Their frontier either side of it, read from their Figure 6, is
# (-0.66 @ 13.69 per cent) and (-1.38 @ 10.08 per cent).
#
# WHERE THIS RUN SHOULD LAND. The control is 13.05 per cent of the model at
# -0.717, or -0.635 on the mean of its last five epochs. T2 at the exposure
# matched knee dose buys about 1.3 points of packet for 0.12 of accuracy, and T3
# at 6e-8 buys a few tenths more at 2.61 points per point. Expect 11.4 to 11.8
# per cent at -0.85 to -0.95. Interpolating NNCodec's own frontier at 11.75 per
# cent gives -0.843, so this run is aimed at a TIE and not at a win. That is the
# honest reading of the exchange rates and it is stated here before the fact.
#
# THE DOSE IS UNCHANGED AND DELIBERATELY SO. T2 stays at 2.1e-7, the exposure
# matched equivalent of the test_231 knee on a thirty-epoch cosine: the schedule
# sums are 10.9 over twenty epochs and 16.2 over thirty, so the 3.16e-7 that
# reproduces the knee on twenty epochs becomes 2.1e-7 here. The control's
# starting sparsity moved only from 16.42 to 15.48 per cent between the two
# recipes, so the knee should sit at the same dose. T3 stays at 6e-8, twice the
# test_234 dose, with the relative dual step that made the channel open at all.
#
# WHY IT RUNS EVEN THOUGH A TIE IS THE EXPECTATION. The pair (test_260,
# test_262) is the only matched control-and-PRESTO comparison at equal budget we
# will have on this architecture, and the paper makes that comparison for every
# other network. It is also the row that shows the regularizer working on a
# network with no accuracy headroom, which is a different claim from the ResNet
# ones and worth its own line.
#
# COST: 388-397s per epoch measured on test_234, so about three hours and twenty
# minutes for thirty epochs. The six-hour wall is slack.

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
