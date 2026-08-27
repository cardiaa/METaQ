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

# EfficientNet-B0, METaQ sulla ricetta con i nove layer piu' saturi a otto bit.
#
# ULTIMA LEVA RIMASTA SU QUESTA RETE, E VA IN FONDO ALLA LISTA. Il test_260 ha
# alzato il tetto di 0.36 punti portando a otto bit i sedici depthwise. Nel suo
# log restano nove layer con clipping alto -- squeeze-excite fc2 e convoluzioni
# di espansione 1x1 -- che insieme sono lo 0.15 per cento dei pesi e costerebbero
# 0.019 punti di pacchetto, praticamente nulla.
#
# PERCHE' SERVIREBBE, in aritmetica. NNCodec consegna 76.78 all'11.25 per cento
# del modello. Il nostro controllo per-tensore sta a 76.811, cioe' 0.031 sopra:
# quello e' TUTTO il margine che abbiamo, e scendere a 10.36 per cento di
# pacchetto costa fra 0.05 e 0.09 punti anche al miglior tasso di cambio mai
# misurato su questa rete. Quindi al loro punto operativo non ci arriviamo, e non
# perche' la dose sia sbagliata ma perche' il tetto e' troppo basso. Se questi
# nove layer valessero altri 0.15-0.20 punti il controllo salirebbe verso 77.0 e
# il conto tornerebbe: da li' scendere costa 0.09 e si atterra sopra il loro
# 76.78 restando piu' piccoli.
#
# PERCHE' E' SPECULATIVO. Non c'e' evidenza che questi nove valgano quanto i
# depthwise: quelli avevano un meccanismo preciso dietro, un canale di uscita per
# un canale di ingresso e nessuna media su una somma, mentre questi sono
# semplicemente i piu' saturi rimasti. Il clipping medio pesato per parametri nel
# test_260 e' gia' 1.01 per cento, cioe' sano.
#
# COSA CAMBIA DAL RUN PRECEDENTE: solo --layer_C, da diciassette tensori a otto
# bit a ventisei. Tutto il resto identico.
#
# COSTO: 370s per epoca, circa tre ore.

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
        --layer_C 256,256,256,16,256,256,256,16,256,256,256,256,16,256,16,256,256,16,256,16,16,256,16,16,16,16,256,16,16,16,16,256,16,16,16,16,256,16,16,16,16,256,16,16,16,16,256,16,16,16,16,256,16,16,16,16,256,16,16,16,16,256,16,16,16,16,256,16,16,16,16,256,16,16,16,16,256,16,16,16,16,16 \
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
        --lsq_per_channel N \
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
