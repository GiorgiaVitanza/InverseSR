#!/bin/bash

#SBATCH --job-name=vae_global_arcsinh_scheduler
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --account=IscrC_DATIV-ML
#SBATCH --output=vae_local_val_Data_mix_OK_%j.out
#SBATCH --error=vae_local_val_Data_mix_OK_%j.err
#SBATCH --mem=160G


module purge

module load cuda/12.2          
module load python/3.11.7

SCRATCH=/leonardo_scratch/large/userexternal/gvitanza/InverseSR
HOME=/leonardo/home/userexternal/gvitanza/

source ${HOME}/.venv/bin/activate

DATASET=Data_mix_OK
EPOCHS=300
Z_CHANNELS=3
NORM_MODE='local'
EXPERIMENT_NAME=${EPOCHS}_z${Z_CHANNELS}_${NORM_MODE}_${DATASET}_val

python3 ${SCRATCH}/project/ml_flow_train_vae_decoder.py\
    --data_dir "${SCRATCH}/data/inputs/${DATASET}"\
    --tensor_board_logger_vae "${SCRATCH}/logs_vae/vae_decoder_${EXPERIMENT_NAME}"\
    --output_dir_vae "${SCRATCH}/data/trained_models_astro/vae_decoder_${EXPERIMENT_NAME}"\
    --z_channels $Z_CHANNELS\
    --norm_mode $NORM_MODE\
    --learning_rate 1e-4\
    --epochs $EPOCHS\