#!/bin/bash
#SBATCH --job-name=VAE_astro          # Nome del job
#SBATCH --account=IscrC_DATIV-ML
#SBATCH --partition=boost_usr_prod     # Partizione per le GPU
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1                      # Usiamo 1 nodo
#SBATCH --ntasks-per-node=1            # Un solo task principale
#SBATCH --gres=gpu:1                   # Chiediamo 1 GPU A100 (puoi metterne fino a 4)
#SBATCH --cpus-per-task=8  		# Core CPU per il dataloading
#SBATCH --mem=32GB                     # Memoria RAM
#SBATCH --time=4-00:00:00
#SBATCH --output=vae_decoder_10_1ch_z16%j.out           # File dove finiranno i print dello script
#SBATCH --error=vae_decoder_10_1ch_z16%j.err

# 1. Carica i moduli necessari (Leonardo usa LMOD)
# Sostituisci le righe del module load con queste:
module purge
module load profile/deeplrn
module load python/3.11.7


source /leonardo_scratch/large/userexternal/gvitanza/InverseSr-Astro/.venv/bin/activate

EPOCHS=10

# Lancio del training
python /leonardo_scratch/large/userexternal/gvitanza/InverseSR/project/ml_flow_train_vae_decoder.py --batch_size 2 --epochs $EPOCHS --data_dir "./data/inputs/128x128x128_stride128/npy_patches"  --output_dir "./data/trained_models_astro/vae_decoder_train_1ch_$EPOCHS" --catalogue_path "./data/inputs/128x128x128_stride128/train_catalog.csv" --resolution 128 128 128

echo "Job completed."

