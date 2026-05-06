#!/bin/bash
#SBATCH --job-name=ddpm_train          # Nome del job
#SBATCH --account=IscrC_DATIV-ML
#SBATCH --partition=boost_usr_prod     # Partizione per le GPU
#SBATCH --nodes=1                      # Usiamo 1 nodo
#SBATCH --ntasks-per-node=1            # Un solo task principale
#SBATCH --gres=gpu:1                   # Chiediamo 1 GPU A100 (puoi metterne fino a 4)
#SBATCH --cpus-per-task=8          # Core CPU per il dataloading
#SBATCH --mem=32GB                     # Memoria RAM
#SBATCH --time=24:00:00                # Tempo massimo (HH:MM:SS)
#SBATCH --output=ddpmv3_10_crossattn_z8_global_new_%j.out           # File dove finiranno i print dello script
#SBATCH --error=ddpmv3_10_crossattn_z8_global_new_%j.err

# 1. Carica i moduli necessari 
module purge
module load profile/deeplrn
module load python/3.11.7

source /leonardo_scratch/large/userexternal/gvitanza/InverseSR/.venv/bin/activate

EPOCHS=10
BATCH_SIZE=2
IN_CHANNELS_UNET=8
NORM_MODE="global_sym"

# Lancio del training
python project/ml_flow_train_ddpm_v3.py \
--in_channels_unet $IN_CHANNELS_UNET \
--out_channels_unet $IN_CHANNELS_UNET \
--data_dir "./data/inputs/128x128x128_stride128/train/npy_patches" \
--catalogue_path "./data/inputs/128x128x128_stride128/train/train_catalog.csv" \
--use_spatial_transformer \
--output_dir_ddpm "./data/trained_models_astro/ddpm_cross_attn_${EPOCHS}_${BATCH_SIZE}_z${IN_CHANNELS_UNET}_${NORM_MODE}_new" \
--epochs $EPOCHS \
--batch_size $BATCH_SIZE \
--vae_path "./checkpoints_vae_decoder_${IN_CHANNELS_UNET}_10epochs_May04_19-02-45/vae_full_ep10.pth" \
--z_channels $IN_CHANNELS_UNET \
--cond_key "crossattn" \
--context_dim 4 \
--image_size 32 \
--norm_mode $NORM_MODE \

echo "Job completed."

