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
#SBATCH --output=ddpmv3_20_crossattn_z16%j.out           # File dove finiranno i print dello script
#SBATCH --error=ddpmv3_20_crossattn_z16%j.err

# 1. Carica i moduli necessari 
module purge
module load profile/deeplrn
module load python/3.11.7


cd /leonardo_scratch/large/userexternal/gvitanza/InverseSr-Astro/
source .venv/bin/activate



# Lancio del training
python project/ml_flow_train_ddpm_v3.py --in_channels_unet 16 --data_dir "./data/inputs/128x128x128_stride128/npy_patches" --catalogue_path "./data/inputs/128x128x128_stride128/train_catalog.csv"  --use_spatial_transformer --output_dir "./data/outputs/ddpm_cross_attn_20_2_z16" --epochs 20 --batch_size 2 --vae_path "./data/trained_models_astro/vae_decoder_train_1ch_10/checkpoints_vae_decoder_1_10epochs/vae_full_ep10.pth" --cond_key "crossattn" --context_dim 4

echo "Job completed."

