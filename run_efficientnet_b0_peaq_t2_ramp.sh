#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_ObCTDoNN
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=03:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# test_231: EfficientNet-B0, T2 dose-response curve in a single job. A twenty
# epoch ramp on a deliberately over-dosed T2, which writes one frontier row per
# epoch. This is the test_228 trick, and it is the only method that has ever
# produced a usable T2 dose in this project: three analytic derivations in a row
# were wrong on AlexNet (test_226 inert, test_227 runaway by 3600x, test_228
# over-dosed by ten) and the number that finally worked was READ OFF a ramp.
#
# BASELINE: test_230, same recipe with every METaQ coefficient at zero. A_Q
# 76.590 against 77.566 FP32 (-0.976), packet 11.55%, 16.19% sparsity emerging
# from the LSQ zero bin alone, clipping mean 2.70%, step sizes within 2% of
# their init on 75 of 82 layers. Every row this run produces is to be read
# against test_230 AT THE SAME EPOCH, exactly as test_228/229 were read against
# test_225. The baseline is not lossless and that is deliberate: the frontier
# (accuracy lost per unit of compression gained) is a difference, and a
# difference does not need its origin to be at zero.
#
# DOSE. T2 enters as a scalar --sparsity_coeff, uniform across all 82 quantized
# tensors, exactly as test_168 did on ResNet-18. Not --layerwise_t2_targets:
# that path needs 82 target sparsities and a displacement calibration whose only
# validation is on AlexNet, which is a lot of machinery to trust on a network
# whose dose is unknown. The cost of the scalar is that the force T2/(q_edge*a)
# varies about sevenfold across layers, since EfficientNet's step sizes span
# 0.030 to 0.207. test_168 accepted the same and came out lossless.
#
# TOP OF THE RAMP: 4e-6. Anchored on test_168, where T2=1e-7 with T1=1e-5 bought
# 47.57% sparsity on ResNet-18, then carried over by the two ratios that the
# floor-regime force actually depends on:
#   - step size, median 0.0351 here against 0.0160 on ResNet-18. The force is
#     T2/(q_edge*a), and the distance to cover is a/2, so a enters twice: 4.8x.
#   - displacement budget per epoch, lr*steps = 6.24 here against 12.51 there,
#     since global batch 4096 buys speed by giving up steps: 2x.
# which puts the ~47% dose near 1e-6. The ramp tops out four times above that on
# purpose, so the sweep runs from clearly inert to clearly over-compressed and
# the interesting band lands around epoch five rather than at the far end. If
# the estimate is off by five in either direction the ramp still crosses the
# frontier, which is the whole point of tracing instead of guessing.
#
# T1 GOES BACK ON at 1e-5. The perspective inner problem needs it: with T1=0 the
# ridge T1*w^2/y vanishes and y* stops being well posed, so T2 alone is not a
# configuration the solver has. This re-introduces the risk test_230 was built
# to avoid, and the check is now cheap because the baseline measured it: if the
# clipped fraction climbs above test_230's 2.70% mean, or if the depthwise
# layers start walking the way features.7.0.block.1.0 did in test_222 (9.3% ->
# 34.7%), that is the T1 feedback loop and not T2 working. Note the loop needed
# T2=T3=0 to exist at all: with T2 live the grid geometry carries compression
# information and the full envelope derivative is the correct one, so the
# expectation is that it does NOT return.
#
# --min_lr 1e-3, i.e. the same 5% floor test_228 used at its own base rate. The
# phase schedule cosines the rate down along the ramp, so without a floor the
# late rows -- the ones at the doses that matter -- would be measured on a
# network too frozen to defend itself, and would read as worse than they are.
#
# HOW TO READ IT, and when to kill it. One row per epoch, sparsity and packet
# against A_Q, and the deficit against test_230's same epoch. Expect the knee:
# on AlexNet it sat between 42% and 51% sparsity and the deficit tripled across
# it. If sparsity is past 80% by epoch six the top was too high, and the useful
# rows are already in the log -- kill the job and keep them. If sparsity has not
# left the baseline's 16% by epoch ten the top was too low, kill it and multiply
# by ten; that costs half an hour and no interpretation.
#
# COST. test_230 ran 157s/epoch. T2 with T3 off goes through the cheap per-step
# path, which allocates once per layer per step and there are 82 layers, so
# expect 200-220s/epoch and about an hour and ten for twenty epochs.
#
# AND THE POINT OF ALL THIS: at 312 steps per epoch, T3 costs 78 dual calls per
# epoch here against the 313 it would cost on AlexNet. The complete T2+T3 run
# that was fourteen hours on AlexNet lands around ninety minutes on this
# network, and sixty epochs stay affordable if the frontier says they are worth
# it. That was the real return on the batch-256 decision.

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
        --n_epochs 20 \
        --diagnostic_epochs 0 \
        --metaq_ramp_epochs 20 \
        --metaq_flat_epochs 0 \
        --lr 2e-2 \
        --lr_warmup_epochs 2 \
        --min_lr 1e-3 \
        --optimizer_weight_decay 1e-4 \
        --perspective_coeff 1e-5 \
        --entropy_coeff 0 \
        --sparsity_coeff 4e-6 \
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
        --entropy_warmup_epochs 0 \
        --entropy_every 4 \
        --dual_step 3e-9 \
        --check_ddp_sync \
        --pretrained Y \
        > "$OUTPUT_TARGET" 2>&1
'
