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

# test_224: AlexNet, LSQ-only lossless baseline, second attempt.
#
# test_223 (same recipe, 20 epochs, uniform C=16) landed at A_Q 55.956 against
# 56.524 FP32, i.e. -0.568, against -1.644 for the best previous AlexNet
# (test_204).  Two thirds of the gap closed, but not lossless.  The log says the
# run was stopped by its budget, not by the method:
#   - A_Q rises monotonically over the last eight epochs, 55.250 -> 55.956, and
#     the FINAL epoch still gains +0.098;
#   - level_change decays 6.23% -> 0.143% as the cosine kills the rate, so the
#     reassignment mechanism switched off while the accuracy was still climbing.
#     Exactly the shape of test_168 on ResNet-18, which was also still rising at
#     the end.
#   - no layer is in a pathological regime: clipping is flat between 0.5% and
#     1.0% everywhere and every step size is stable within 8% of its init, with
#     one exception, features.0, whose clipped fraction nearly doubled
#     (1.29% -> 2.37%) while its step size shrank to 0.893.
#   - metaq_scale_grad_last stayed exactly zero for all twenty epochs: the T1
#     gate holds.
#
# So this run changes exactly two things.
#   a) --n_epochs 20 -> 40.  The cosine now traverses the productive band of the
#      schedule over twice as many steps.  The whole +0.71 of test_223 was
#      earned between epoch 10 and epoch 20, i.e. while the rate fell from
#      1.1e-3 to the floor; doubling the length doubles the time spent there and
#      still anneals properly at the end, which a constant floor would not.
#   b) --layer_C 256,256,16,16,16,16,16,16: the two input convolutions at 8 bits.
#      features.0 is the only layer showing a runaway, and it is 11x11x3 with a
#      very wide per-filter dynamic range; features.3 is the next most fragile.
#      Together they are 330432 parameters, 0.54% of the network, so the extra
#      four bits cost 0.068% of the FP32 size in nominal terms and less after
#      entropy coding.  This is also what Deep Compression does on AlexNet.
#
# Original test_223 header follows.
#
# AlexNet, LSQ-only lossless baseline. First run of the sweep with the
# corrected training recipe, and the reference point that every later AlexNet run
# (T2 on, then T2 and T3 on) must reproduce exactly except for the coefficients.
#
# What changes with respect to test_205, and why.
#
# 1. lr 1e-4 -> 2e-3 and global batch 4096 -> 1024.  1e-4 was the DeiT-Small
#    ADAM rate applied to SGD.  Over the ten epochs of test_205 the AlexNet
#    weight distribution moved by 0.02% (first percentile -3.174981e-02 ->
#    -3.174318e-02): the total displacement budget was about 0.2 quantization
#    steps for the whole run, so not one weight could be reassigned to a
#    different level and the deployed model stayed the post-training
#    quantization of the checkpoint.  2e-3 at global batch 1024 is AlexNet's
#    from-scratch per-sample rate divided by forty, exactly the safety factor of
#    the lossless test_168 on ResNet-18, and the smaller batch buys four times
#    the optimizer steps on the same data.  Expected budget: about fifty
#    quantization steps.  The new qat_progress line in the log measures it
#    directly, epoch by epoch.
# 2. --lr_warmup_epochs 1.  AlexNet has no normalization layers and the rate is
#    twenty times the historical one; the first epoch runs at half rate.
# 3. T1 stays at 1e-5 and is now honest.  Until this commit T1 also fed the LSQ
#    step sizes through the representability boundary, with a strictly positive
#    contribution proportional to the clipped count: on test_222 the step sizes
#    shrank monotonically and one depthwise layer went from 9.3% to 34.7% of its
#    weights pinned at the codebook edge with zero gradient.  With T2 = T3 = 0 the
#    term is now what it is supposed to be, a plain ridge on the latent weights,
#    and the trainer takes a fast path that computes exactly 2*T1*w in one
#    operation (bit-exact against the general path, see
#    scratchpad/verify_lossless_fixes.py).  T1 is held at 1e-5 in ALL THREE runs
#    of the ladder so that only T2 and T3 ever change; confirm the gate in the
#    log by checking that metaq_scale_grad_last is all zeros.
# 4. --lsq_per_channel Y.  conv1 is 11x11x3 with a very wide per-filter dynamic
#    range and a per-tensor step size serves it badly.  The step sizes are paid
#    for honestly in metadata_bits: 10344 of them, 0.017% of the FP32 size.
# 5. --lsq_scale_lr_mode relative with 1e-3.  Adam moves a parameter by about lr
#    in absolute units, so one absolute rate means a fifteen times different
#    relative rate across AlexNet's step sizes.  1e-3 relative reproduces the
#    conditioning of test_168.
# 6. --lsq_scale_lr_schedule Y.  The step sizes now decay with the weights
#    instead of continuing to move after the network has settled.
# 7. --distillation Y --distill_alpha 0.5 from the FP32 teacher.  Worth +0.33
#    points on AlexNet at C=16 (test_204 against test_205) and standard practice
#    in every QAT paper we compare against.
# 8. Corrected MSE step-size initialization (range_margin 0 instead of 0.5): the
#    search now lands on candidate ~30/100 instead of ~4/100 on the fully
#    connected layers.
#
# Everything here except --entropy_coeff and --sparsity_coeff is the fixed
# AlexNet recipe and must not change in the next two runs of the sweep.

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
        --n_epochs 40 \
        --lr 2e-3 \
        --lr_warmup_epochs 1 \
        --optimizer_weight_decay 1e-4 \
        --perspective_coeff 1e-5 \
        --entropy_coeff 0 \
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
        --dual_step 3e-9 \
        --check_ddp_sync \
        --pretrained Y \
        > "$OUTPUT_TARGET" 2>&1
'
