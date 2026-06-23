#!/bin/bash

#SBATCH --job-name=Feature_analysis
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --account=IscrC_DATIV-ML
#SBATCH --output=Features_%j.out
#SBATCH --error=Features_%j.err





module purge

module load cuda/12.2          
module load python/3.11.7

SCRATCH=/leonardo_scratch/large/userexternal/gvitanza/InverseSR

HOME=/leonardo/home/userexternal/gvitanza

source ${HOME}/.venv/bin/activate

python3 $SCRATCH/project/utils/feature_analysis.py