#!/bin/bash
#SBATCH --job-name=astro_recon          # Nome del job
#SBATCH --output=vis_%j.out      # File di log (assicurati che la cartella 'logs' esista)
#SBATCH --error=vis_%j.err       # File di errore
#SBATCH --partition=boost_usr_prod      # Partizione standard per Leonardo Booster
#SBATCH --nodes=1                       # Usiamo 1 nodo
#SBATCH --ntasks-per-node=1             # 1 task (il tuo script python)
#SBATCH --gres=gpu:1                    # Fondamentale: richiede 1 GPU A100
#SBATCH --cpus-per-task=8               # Numero di core CPU (aiuta nel caricamento dati)
#SBATCH --mem=64G                       # RAM di sistema (per evitare il "Killed")
#SBATCH --time=02:00:00                 # Tempo massimo (2 ore, regola in base ai 50 step DDIM)
#SBATCH --account=IscrC_DATIV-ML # Sostituisci con il tuo ID progetto (es. Eur_...)

# 1. Caricamento moduli necessari
module purge
module load cuda
module load python/3.11.7 # Assicurati che coincida con la tua venv


# 2. Attivazione ambiente virtuale
# Sostituisci con il percorso reale della tua venv su Leonardo
source /leonardo_scratch/large/userexternal/gvitanza/InverseSr-Astro/.venv/bin/activate

# 3. Impostazione variabili d'ambiente per PyTorch
export CUDA_VISIBLE_DEVICES=0

# 4. Esecuzione del codice
# Uso 'python -u' per avere l'output non bufferizzato nei log
python -u project/visualizzazione_output.py
