#!/bin/bash

#SBATCH --job-name=test_vae
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --account=IscrC_DATIV-ML
#SBATCH --output=test_vae_%j.out
#SBATCH --error=test_vae_%j.err
#SBATCH --mem=160G


module purge

module load cuda/12.2          
module load python/3.11.7

SCRATCH=/leonardo_scratch/large/userexternal/gvitanza/InverseSR
HOME=/leonardo/home/userexternal/gvitanza

source ${HOME}/.venv/bin/activate

python3 ${SCRATCH}/project/test_vae.py\
    --test_dir "$SCRATCH/data/inputs/128x128x128_stride128/test/npy_patches"\
    --catalogue_path "$SCRATCH/data/inputs/128x128x128_stride128/test/test_catalog.csv" \
    --z_channels 3\
    --norm_mode 'local'\
    --vae_path  "$SCRATCH/vae_decoder_3_100epochs_local_May22_20-59-37/vae_full_ep100.pth"\
    --test_fig "$SCRATCH/data/outputs/test_vae"