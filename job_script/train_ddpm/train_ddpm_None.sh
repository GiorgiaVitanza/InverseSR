#!/bin/bash

#SBATCH --job-name=ddpm_None
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --account=IscrC_DATIV-ML
#SBATCH --output=ddpm_None_datamix_b8_%j.out
#SBATCH --error=ddpm_None_datamix_b8_%j.err


module purge

module load cuda/12.2          
module load python/3.11.7

SCRATCH=/leonardo_scratch/large/userexternal/gvitanza/InverseSR
HOME=/leonardo/home/userexternal/gvitanza/

source ${HOME}/.venv/bin/activate


DATASET=Data_mix_OK
EPOCHS=1000
Z_CHANNELS=3
NORM_MODE='local'
COND='None'
EXPERIMENT_NAME=${COND}_${EPOCHS}_z${Z_CHANNELS}_${NORM_MODE}_${DATASET}_b8

python3 ${SCRATCH}/project/ml_flow_train_ddpm_v3.py\
    --data_dir "${SCRATCH}/data/inputs/${DATASET}"\
    --in_channels_unet $Z_CHANNELS\
    --out_channels_unet $Z_CHANNELS\
    --tensor_board_logger_ddpm "${SCRATCH}/logs_ddpm/ddpm_${EXPERIMENT_NAME}"\
    --output_dir_ddpm "${SCRATCH}/data/trained_models_astro/ddpm_${EXPERIMENT_NAME}"\
    --epochs $EPOCHS\
    --z_channels $Z_CHANNELS\
    --norm_mode $NORM_MODE\
    --learning_rate 1e-4\
    --cond_key $COND\
    --vae_path "${SCRATCH}/vae_decoder_3_300epochs_${NORM_MODE}_Oct02_10-21-09/vae_full_ep300.pth"\
    --no-use_mask_channel \
    --no-use_spatial_transformer \
    --batch_size 8\
