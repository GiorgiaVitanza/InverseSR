#!/bin/bash
#SBATCH --job-name=ddpm_astro          # Nome del job
#SBATCH --account=IscrC_DATIV-ML
#SBATCH --partition=boost_usr_prod     # Partizione per le GPU
#SBATCH --nodes=1                      # Usiamo 1 nodo
#SBATCH --ntasks-per-node=1            # Un solo task principale
#SBATCH --gres=gpu:1                   # Chiediamo 1 GPU A100 (puoi metterne fino a 4)
#SBATCH --cpus-per-task=8              # Core CPU per il dataloading
#SBATCH --mem=32GB                     # Memoria RAM
#SBATCH --time=24:00:00                # Tempo massimo (HH:MM:SS)
#SBATCH --output=ddpm_%j.out           # File dove finiranno i print dello script

# 1. Carica i moduli necessari (Leonardo usa LMOD)
# Sostituisci le righe del module load con queste:
module purge
module load profile/deeplrn
module load python/3.11.7


cd /leonardo_scratch/large/userexternal/gvitanza/InverseSr-Astro/
source .venv/bin/activate



# Lancio del training
python project/ml_flow_train_ddpm.py


echo "Job completed."
