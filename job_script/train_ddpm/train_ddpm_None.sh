#!/bin/bash

#SBATCH --job-name=ddpm_None
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --account=IscrC_DATIV-ML
#SBATCH --output=ddpm_None_%j.out
#SBATCH --error=ddpm_None_%j.err


module purge

module load cuda/12.2          
module load python/3.11.7

SCRATCH=/leonardo_scratch/large/userexternal/gvitanza/InverseSR
HOME=/leonardo/home/userexternal/gvitanza/

source ${HOME}/.venv/bin/activate

EPOCHS=100
Z_CHANNELS=3
NORM_MODE='local'
COND='None'

python3 ${SCRATCH}/project/ml_flow_train_ddpm_v3.py\
    --data_dir "${SCRATCH}/ska_hi_dataset/hr"\
    --catalogue_path "${SCRATCH}/ska_hi_dataset/global_catalog.csv"\
    --in_channels_unet $Z_CHANNELS\
    --out_channels_unet $Z_CHANNELS\
    --tensor_board_logger_ddpm "${SCRATCH}/logs_ddpm/ddpm_${COND}_${EPOCHS}_z${Z_CHANNELS}_${NORM_MODE}"\
    --output_dir_ddpm "${SCRATCH}/data/trained_models_astro/ddpm_${COND}_${EPOCHS}_z${Z_CHANNELS}_${NORM_MODE}"\
    --epochs $EPOCHS\
    --z_channels $Z_CHANNELS\
    --norm_mode $NORM_MODE\
    --learning_rate 1e-4\
    --cond_key $COND\
    --vae_path "${SCRATCH}/vae_decoder_3_100epochs_local_Jul08_10-50-02/vae_full_ep50.pth"
