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

# test_256: ViT-B/16, METaQ on the test_251 recipe, twenty epochs. Its matched
# control is test_255, which is the same script with the three coefficients set
# to zero; the two are launched together because runs 250 to 254 have already
# established that the recipe trains this network.
#
# READ THE HEADER LINE FIRST. This is the first ViT run on the corrected tensor
# set. torchvision keeps the packed query/key/value projection of
# nn.MultiheadAttention in a bare Parameter called in_proj_weight, which the old
# ".weight" predicate skipped, so tests 250 to 254 quantized and measured only
# 75.15 per cent of the model. Expect
#   [QUANTIZED TENSORS] count=50, parameters=86292480, fraction_of_model=99.68%
# and stop the run if it still says 38 and 75 per cent: it means the cluster
# checkout has not pulled the fix.
#
# COEFFICIENTS FROM DeiT-Small, the closest network we have calibrated: T1 1e-5,
# T2 5e-8, T3 1.5e-8, the knee of the test_194 frontier. A transfer guess, and
# absolute coefficients have not transferred between distant architectures
# before, so treat the dose as provisional and read the frontier, not the point.
#
# WHERE THE ROOM IS. The control finishes with the package split as mask 2.19
# per cent and values 8.11 per cent, and the surviving weights still carry 3.37
# bits each out of the four the codebook allows. That is more slack than
# EfficientNet-B0 had at 2.41 and far more than AlexNet at 1.12, where T2 had
# already done T3's work. T3 is the term with room here, as on both ResNets.
#
# --dual_step 1.3e-2 relative, and this part does transfer exactly: in relative
# mode the convergence rate is dual_step * (1/C) * N_dual * calls_per_epoch, and
# at batch 64 this network has 1251 steps and entropy_every 4, hence the same 313
# calls as ResNet-18 and ResNet-50.
#
# COST: the control measured 198s per epoch at 75 per cent coverage, so the base
# epoch here should be 210-230s. The dual is the rest: it is dominated by
# per-tensor overhead, 0.93s per call on ResNet-18's 21 tensors and 2.7s on
# ResNet-50's 54, and this network now has 50, so budget roughly 800s of dual on
# top and 1000-1100s per epoch. Twenty epochs is about six hours and the
# twelve-hour wall is slack.
#
# WHAT TO COMPARE AGAINST. NNCodec, Table 1 and Figure 7 of the ICML 2023
# workshop paper: ViT-B/16 at 346.27MB and 81.07 top-1, with a Pareto frontier
# of 80.92 at 14.94 per cent, 80.81 at 12.10, 80.41 at 10.44, 79.95 at 9.49,
# 78.26 at 8.24 and 76.93 at 5.97. Their numbers cover the whole model, which is
# exactly why the accounting above had to be fixed before this run.

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
    echo "DeiT-Small environment not found: $METAQ_PYTHON" > "$LOG_FILE"
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
    --model_name ViT-B-16 \
    --delta -10 \
    --gamma 2 \
    --data_root /leonardo_work/IscrC_ObCTDoNN/acardia0/datasets \
    --train_workers 4 \
    --val_workers 2 \
    --batch_size 64 \
    --n_epochs 20 \
    --lr 5e-5 \
    --optimizer_weight_decay 0 \
    --perspective_coeff 1e-5 \
    --entropy_coeff 1.5e-8 \
    --sparsity_coeff 5e-8 \
    --perspective Y \
    --flat_schedule N \
    --mag_prune_ratio 0 \
    --quantization Y \
    --quantizer lsq \
    --lsq_scale_lr 1e-5 \
    --lsq_init mse \
    --lsq_grad_scaling N \
    --lsq_scale_lr_schedule Y \
    --min_lr 2.5e-6 \
    --joint_lsq_metaq Y \
    --distillation Y \
    --distill_alpha 0.9 \
    --distill_tau 1.0 \
    --bn_recalibration_batches 0 \
    --C 16 \
    --max_iterations 6 \
    --metrics_interval 1 \
    --entropy_warmup_epochs 1 \
    --entropy_every 4 \
    --dual_step 1.3e-2 \
    --dual_step_mode relative \
    --check_ddp_sync \
    --pretrained Y \
    > "$OUTPUT_TARGET" 2>&1
'
