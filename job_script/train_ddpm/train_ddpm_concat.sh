#!/bin/bash

#SBATCH --job-name=ddpm_concat
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --account=IscrC_DATIV-ML
#SBATCH --output=ddpm_concat_new_%j.out
#SBATCH --error=ddpm_concat_new_%j.err


module purge

module load cuda/12.2          
module load python/3.11.7

SCRATCH=/leonardo_scratch/large/userexternal/gvitanza/InverseSR
HOME=/leonardo/home/userexternal/gvitanza

source ${HOME}/.venv/bin/activate

EPOCHS=2
NORM_MODE='local'
Z_CHANNELS=3
IN_UNET_CHANNELS=7
OUT_UNET_CHANNELS=3
COND='concat'

python3 ${SCRATCH}/project/ml_flow_train_ddpm_v3.py\
    --data_dir "${SCRATCH}/data/inputs/128x128x128_stride128/train/npy_patches"\
    --catalogue_path "${SCRATCH}/data/inputs/128x128x128_stride128/train/train_catalog.csv"\
    --in_channels_unet $IN_UNET_CHANNELS\
    --out_channels_unet $OUT_UNET_CHANNELS\
    --tensor_board_logger_ddpm "${SCRATCH}/logs_ddpm/ddpm_${COND}_${EPOCHS}_z${Z_CHANNELS}_${NORM_MODE}"\
    --output_dir_ddpm "${SCRATCH}/data/trained_models_astro/ddpm_${COND}_${EPOCHS}_z${Z_CHANNELS}_${NORM_MODE}"\
    --epochs $EPOCHS\
    --z_channels $Z_CHANNELS\
    --norm_mode $NORM_MODE\
    --learning_rate 1e-4\
    --cond_key $COND\
    --vae_path "${SCRATCH}/vae_decoder_3_100epochs_local_May22_20-59-37/vae_full_ep100.pth"
