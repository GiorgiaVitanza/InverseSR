#!/bin/bash

#SBATCH --job-name=test_ddpm
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --account=IscrC_DATIV-ML
#SBATCH --output=test_ddpm_%j.out
#SBATCH --error=test_ddpm_%j.err
#SBATCH --mem=160G


module purge

module load cuda/12.2          
module load python/3.11.7
SCRATCH=/leonardo_scratch/large/userexternal/gvitanza/InverseSR
HOME=/leonardo/home/userexternal/gvitanza/

source ${HOME}/.venv/bin/activate

python3 ${SCRATCH}/project/test_ddpm.py\
    --test_dir $SCRATCH/data/inputs/16x128x128_stride128_cont_dev/test/npy_patches\
    --catalogue_path $SCRATCH/data/inputs/16x128x128_stride128_cont_dev/test/test_catalog.csv \
    --in_channels_unet 4\
    --out_channels_unet 3\
    --z_channels 3\
    --norm_data local\
    --norm_mode local\
    --cond_key hybrid\
    --vae_path  /leonardo_scratch/large/userexternal/gvitanza/InverseSR/vae_decoder_3_100epochs_local_Jul17_16-52-58/vae_full_ep100.pth\
    --output_dir_ddpm /leonardo_scratch/large/userexternal/gvitanza/InverseSR/ddpm_hybrid_3_100epochs_local_Jul29_11-49-25/ddpm_ep100.pth \
    --test_fig test_ddpm_cont_dev_hybrid\
    --use_mask_channel \
    --use_spatial_transformer \
    --context_dim 4 \
    --epoch 100\