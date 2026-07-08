#!/bin/bash

#SBATCH --job-name=ddpm_None
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --account=IscrC_DATIV-ML
#SBATCH --output=ddpm_None_PROVA_4_%j.out
#SBATCH --error=ddpm_None_PROVA_4_%j.err


module purge

module load cuda/12.2          
module load python/3.11.7

SCRATCH=/leonardo_scratch/large/userexternal/gvitanza/InverseSR

source /leonardo/home/userexternal/gvitanza/.venv/bin/activate


EPOCHS=100
Z_CHANNELS=3
NORM_MODE='local'
COND='None'

python3 ${SCRATCH}/project/ml_flow_train_ddpm_v3.py\
    --catalogue_path "${SCRATCH}/data/inputs/128x128x128_stride128/train/train_catalog.csv"\
    --in_channels_unet $Z_CHANNELS\
    --out_channels_unet $Z_CHANNELS\
    --tensor_board_logger_ddpm "${SCRATCH}/logs_ddpm/ddpm_PROVA_4"\
    --output_dir_ddpm "${SCRATCH}/data/trained_models_astro/ddpm_PROVA_4"\
    --epochs $EPOCHS\
    --z_channels $Z_CHANNELS\
    --norm_mode $NORM_MODE\
    --learning_rate 1e-4\
    --cond_key $COND\
    --vae_path "${SCRATCH}/vae_decoder_3_100epochs_local_Jul07_17-55-43/vae_full_ep10.pth"\
    --image_size 256 256 256\
    --scale_factor 4\
    --batch_size 2\
