#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_ObCTDoNN
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=05:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# test_271: DeiT-Small, METaQ con duale relativo, punto 4, dose doppia.
#
# PERCHE' SERVONO DUE PUNTI IN PIU'. Il blocco 265-267 ha mostrato che la
# frontiera DeiT col duale relativo e' molto piu' piatta di quanto la dose
# trasferita da ViT-B/16 lasciasse pensare:
#   test_265  controllo (0,0,0)          79.706   11.99% del modello
#   test_266  un quinto (1e-8, 3e-9)     79.694   11.98%   <- non comprime NULLA
#   test_267  meta'     (2.5e-8, 7.5e-9) 79.704   10.63%
# Il punto a un quinto ha alzato la sparsita' dal 13.66 al 21.54 per cento senza
# muovere il pacchetto di un decimo: sotto il 50 per cento di sparsita' il costo
# della maschera cresce e mangia esattamente quello che i valori risparmiano. E
# il punto a meta' dose ha tolto 1.36 punti di pacchetto per DUE MILLESIMI di
# accuratezza. Non siamo vicini al ginocchio: siamo ancora nel tratto in cui la
# compressione e' gratis, e una frontiera fatta di tre punti indistinguibili in
# accuratezza non e' una frontiera.
#
# LA DOSE ORIGINALE ERA GIUSTA. Il test_194, con la stessa coppia (5e-8, 1.5e-8)
# ma il duale ASSOLUTO, arrivava al 9.04 per cento di pacchetto. Avevo previsto
# che il duale relativo avrebbe reso il termine entropico molto piu' forte e che
# la dose andasse abbassata: su ViT-B/16 quella previsione ha centrato il
# ginocchio al primo colpo, su DeiT-Small e' stata sbagliata. Questi due run
# tornano alla dose piena e la raddoppiano.

LOG_DIR=$WORK/acardia0/LeonardoTests
mkdir -p "$LOG_DIR"

# NUMERAZIONE FISSA. Gli script di questo blocco partono insieme e
# l'auto-incremento li farebbe collidere sullo stesso file di log.
NEXT=271
LOG_FILE=$LOG_DIR/Leonardo_test_${NEXT}.log
if [ -e "$LOG_FILE" ]; then
    echo "Leonardo_test_271.log esiste gia': rifiuto di sovrascriverlo." >&2
    exit 1
fi
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
    --model_name DeiT-Small \
    --delta -10 \
    --gamma 2 \
    --data_root /leonardo_work/IscrC_ObCTDoNN/acardia0/datasets \
    --train_workers 4 \
    --val_workers 2 \
    --batch_size 128 \
    --n_epochs 25 \
    --lr 1e-4 \
    --optimizer_weight_decay 0 \
    --perspective_coeff 1e-5 \
    --entropy_coeff 3e-8 \
    --sparsity_coeff 1e-7 \
    --perspective Y \
    --flat_schedule N \
    --mag_prune_ratio 0 \
    --quantization Y \
    --quantizer lsq \
    --lsq_scale_lr 1e-5 \
    --lsq_init mse \
    --lsq_grad_scaling N \
    --lsq_scale_lr_schedule Y \
    --min_lr 5e-6 \
    --joint_lsq_metaq Y \
    --distillation Y \
    --distill_alpha 0.9 \
    --distill_tau 1.0 \
    --bn_recalibration_batches 0 \
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
