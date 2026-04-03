#!/bin/bash
#SBATCH --job-name=VAE_astro
#SBATCH --account=IscrC_DATIV-ML
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=4-00:00:00
#SBATCH --output=vae_decoder_%j.out
#SBATCH --error=vae_decoder_%j.err

# 1. Carica i moduli
module purge
module load profile/deeplrn
module load python/3.11.7

# 2. Attiva l'ambiente
source /leonardo_scratch/large/userexternal/gvitanza/InverseSR/.venv/bin/activate

# 3. Variabili
EPOCHS=10
BASE_DIR="/leonardo_scratch/large/userexternal/gvitanza/InverseSR"
Z_CHANNELS=8

# 4. Lancio del training

python ${BASE_DIR}/project/ml_flow_train_vae_decoder.py \
    --data_dir "${BASE_DIR}/data/inputs/128x128x128_stride128/npy_patches" \
    --output_dir_vae "${BASE_DIR}/data/trained_models_astro/vae_decoder_train_${EPOCHS}ep_z${Z_CHANNELS}" \
    --catalogue_path "${BASE_DIR}/data/inputs/128x128x128_stride128/train_catalog.csv" \
    --batch_size 2 \
    --learning_rate 1e-5 \
    --epochs $EPOCHS \
    --resolution 128 128 128 \
    --z_channels $Z_CHANNELS

echo "Job completed."