#!/bin/bash

#SBATCH --job-name=train_vae
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --account=IscrC_DATIV-ML
#SBATCH --output=vae_%j.out
#SBATCH --error=vae_%j.err
#SBATCH --mem=160G


module purge

module load cuda/12.2          
module load python/3.11.7

SCRATCH=/leonardo_scratch/large/userexternal/gvitanza/InverseSR
HOME=/leonardo/home/userexternal/gvitanza/

source ${HOME}/.venv/bin/activate

EPOCHS=100
Z_CHANNELS=3
NORM_MODE='local'

python3 ${SCRATCH}/project/ml_flow_train_vae_decoder.py\
    --data_dir "${SCRATCH}/ska_hi_dataset/hr"\
    --catalogue_path "${SCRATCH}/ska_hi_dataset/global_catalog.csv"\
    --tensor_board_logger_vae "${SCRATCH}/logs_vae/vae_decoder_${EPOCHS}_z${Z_CHANNELS}_${NORM_MODE}"\
    --output_dir_vae "${SCRATCH}/data/trained_models_astro/vae_decoder_${EPOCHS}_z${Z_CHANNELS}_${NORM_MODE}"\
    --epochs $EPOCHS\
    --z_channels $Z_CHANNELS\
    --norm_mode $NORM_MODE\
    --learning_rate 1e-4\