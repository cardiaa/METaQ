# Usa un'immagine base con Python e PyTorch per CPU
FROM python:3.9

# Imposta la directory di lavoro
WORKDIR /workspace

# Copia tutto il codice dentro il container
COPY . .

# Installa le dipendenze
RUN pip install torch torchvision numpy zstandard

# Comando di default (puoi cambiarlo quando lanci il container)
CMD ["python", "train.py"]
