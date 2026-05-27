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

source ${SCRATCH}/.venv/bin/activate

python3 ${SCRATCH}/project/test_ddpm.py\
    --data_dir $SCRATCH/data/inputs/128x128x128_stride128/test/npy_patches\
    --catalogue_path $SCRATCH/data/inputs/128x128x128_stride128/test/test_catalog.csv \
    --in_channels_unet 3\
    --out_channels_unet 3\
    --z_channels 3\
    --cond_key None\
    --vae_path  /leonardo_scratch/large/userexternal/gvitanza/InverseSR/vae_decoder_3_100epochs_local_May22_20-59-37/vae_full_ep100.pth\
    --output_dir_ddpm /leonardo_scratch/large/userexternal/gvitanza/InverseSR/ddpm_None_3_100epochs_local_May18_11-33-46/ddpm_ep100.pth \
    --test_fig /leonardo_scratch/large/userexternal/gvitanza/InverseSR/data/outputs/test_ddpm_None