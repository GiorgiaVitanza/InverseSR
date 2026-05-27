#!/bin/bash

#SBATCH --job-name=visualizzazione
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --account=IscrC_DATIV-ML
#SBATCH --output=visual_%j.out
#SBATCH --error=visual_%j.err


module purge

module load cuda/12.2          
module load python/3.11.7

SCRATCH=/leonardo_scratch/large/userexternal/gvitanza/InverseSR

source ${SCRATCH}/.venv/bin/activate

python3 $SCRATCH/project/visualizzazione_output.py\
    --image_size 32 32 32\
    --z_channels 3\