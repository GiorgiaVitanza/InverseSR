#!/bin/bash
#SBATCH --job-name=ddpm_train          # Nome del job
#SBATCH --account=IscrC_DATIV-ML
#SBATCH --partition=boost_usr_prod     # Partizione per le GPU
#SBATCH --nodes=1                      # Usiamo 1 nodo
#SBATCH --ntasks-per-node=1            # Un solo task principale
#SBATCH --gres=gpu:1                   # Chiediamo 1 GPU A100 (puoi metterne fino a 4)
#SBATCH --cpus-per-task=8              # Core CPU per il dataloading
#SBATCH --mem=32GB                     # Memoria RAM
#SBATCH --time=24:00:00                # Tempo massimo (HH:MM:SS)
#SBATCH --output=ddpmv3_100_None%j.out           # File dove finiranno i print dello script
#SBATCH --error=ddpmv3_100_None%j.err

# 1. Carica i moduli necessari (Leonardo usa LMOD)
# Sostituisci le righe del module load con queste:
module purge
module load profile/deeplrn
module load python/3.11.7


cd /leonardo_scratch/large/userexternal/gvitanza/InverseSr-Astro/
source .venv/bin/activate



# Lancio del training
python project/ml_flow_train_ddpm_v3.py  --cond_key None --data_dir "./data/inputs/128x128x128_stride128/npy_patches" --catalogue_path "./data/inputs/128x128x128_stride128/train_catalog.csv" --output_dir "./data/outputs/ddpm_None_100_2" --epochs 100 --batch_size 2 --vae_path "./data/trained_models_astro_400/vae_full_ep400.pth" 

echo "Job completed."

