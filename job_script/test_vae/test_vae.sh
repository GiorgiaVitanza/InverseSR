#!/bin/bash

#SBATCH --job-name=test_vae
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --account=IscrC_DATIV-ML
#SBATCH --output=test_vae_global1_HI_%j.out
#SBATCH --error=test_vae_global1_HI_%j.err
#SBATCH --mem=160G


module purge

module load cuda/12.2          
module load python/3.11.7

SCRATCH=/leonardo_scratch/large/userexternal/gvitanza/InverseSR
HOME=/leonardo/home/userexternal/gvitanza/

source ${HOME}/.venv/bin/activate
NORM_MODE="global_1"

python3 ${SCRATCH}/project/test_vae.py\
    --test_dir "$SCRATCH/data/inputs/HI_datamix_128x128x128_stride64/test/npy_patches"\
    --z_channels 3\
    --norm_mode $NORM_MODE\
    --vae_path  "$SCRATCH//vae_decoder_3_300epochs_global_1_Oct30_16-14-13/vae_best_model.pth"\
    --test_fig "$SCRATCH/data/outputs_paper/test_vae_HI_${NORM_MODE}_Oct30_best"