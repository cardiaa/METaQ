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

# test_269: ViT-B/16, la riga di punta del paper rifatta con --max_iterations 3.
# UN SOLO FLAG diverso dal test_256: 6 -> 3.
#
# PERCHE'. test_256 e test_259 sono gli unici due run del paper con N_dual = 6;
# ResNet-18, ResNet-50, DeiT-Small e EfficientNet-B0 usano 3. E con il passo
# duale RELATIVO questa non e' una semplice tolleranza di solutore: la velocita'
# di convergenza del duale e' dual_step * N_dual * chiamate_per_epoca / C, quindi
# N_dual entra moltiplicativamente e a 6 il duale converge al doppio della
# velocita' degli altri. E' una differenza di metodo, non di precisione
# numerica, e va verificata o dichiarata.
#
# PERCHE' QUESTO PUNTO E NON IL GINOCCHIO. Il test_256 e' la riga con cui il
# paper apre il capitolo ViT: 80.372 al 4.41 per cento del modello, contro il
# 75.22 che NNCodec consegna al 4.85. Cinque punti di margine sono il risultato
# piu' forte che abbiamo, ed e' quello su cui vale la pena spendere otto ore di
# verifica. La sensibilita' a N_dual e' la stessa sui due punti, perche' il passo
# relativo misura il passo come frazione dell'intervallo duale e quindi la
# velocita' di convergenza non dipende da T3.
#
# COME SI LEGGE L'ESITO. Se atterra entro il rumore del test_256 (la sigma
# misurata sui tre semi di ResNet-18 e' 0.083 sull'accuratezza e 0.015 sul
# pacchetto), l'appendice dichiara che la riga di punta e' stata verificata al
# valore comune del solutore e i tre punti ViT restano quelli che abbiamo. Se
# invece si sposta, vanno rifatti anche il test_255 non serve ma il 259 si', e
# il capitolo ViT costa altre sedici ore.
#
# COSTO: il test_256 ha misurato 2229s per epoca, di cui 218 di forward e 2011 di
# duale. Dimezzando le iterazioni duali il costo per chiamata non si dimezza
# esattamente, perche' la costruzione dell'inviluppo inferiore e' fatta una volta
# per chiamata e non per iterazione, quindi la stima e' 1250-1500s per epoca,
# cioe' fra sette e otto ore e mezza per venti epoche. Il wall di dodici ore
# copre anche il caso peggiore, ed e' comunque meno delle undici ore e cinquanta
# che il test_256 ha impiegato a N_dual 6.

LOG_DIR=$WORK/acardia0/LeonardoTests
mkdir -p "$LOG_DIR"

# NUMERAZIONE FISSA. Gli script di questo blocco partono insieme e
# l'auto-incremento li farebbe collidere sullo stesso file di log.
NEXT=269
LOG_FILE=$LOG_DIR/Leonardo_test_${NEXT}.log
if [ -e "$LOG_FILE" ]; then
    echo "Leonardo_test_269.log esiste gia': rifiuto di sovrascriverlo." >&2
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
