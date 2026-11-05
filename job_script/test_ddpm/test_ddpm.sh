#!/bin/bash

#SBATCH --job-name=test_ddpm
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --account=IscrC_DATIV-ML
#SBATCH --output=test_ddpm_global1_ddpm_vae_best_augm_%j.out
#SBATCH --error=test_ddpm_global1_ddpm_vae_best_augm_%j.err
#SBATCH --mem=160G


module purge

module load cuda/12.2          
module load python/3.11.7
SCRATCH=/leonardo_scratch/large/userexternal/gvitanza/InverseSR
HOME=/leonardo/home/userexternal/gvitanza/

source ${HOME}/.venv/bin/activate

python3 ${SCRATCH}/project/test_ddpm.py\
    --test_dir $SCRATCH/data/inputs/16x128x128_cont_ldev_OK/test_augmented/\
    --in_channels_unet 3\
    --out_channels_unet 3\
    --z_channels 3\
    --norm_data global_1\
    --norm_mode global_1\
    --cond_key None\
    --vae_path  $SCRATCH/vae_decoder_3_300epochs_global_1_Oct19_17-53-09/vae_best_model.pth\
    --output_dir_ddpm $SCRATCH/ddpm_None_3_500epochs_global_1_Oct19_21-01-13/ddpm_best_epoch.pth \
    --test_fig $SCRATCH/data/outputs_paper/test_ddpm_cont_ldev_None_global1_vae_ddpm_best_augm\
    --no-use_mask_channel \
    --no-use_spatial_transformer \
    