#!/bin/bash

#SBATCH --job-name=InverseSR_3D
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --account=IscrC_DATIV-ML
#SBATCH --output=InverseSR_astro_%j.out
#SBATCH --error=InverseSR_astro_%j.err


# Variabili utili per Python
export PYTHONUNBUFFERED=1

# ==============================================================================
# SETUP AMBIENTE E MODULI
# ==============================================================================
module purge
# Load necessary modules (adjust to your environment)
module load cuda/12.2
module load python/3.11.7

source $SCRATCH/InverseSr-Astro/.venv/bin/activate
echo -e '\n\n\n'
echo "$(date +"%T"):  start running model!"




python3 $SCRATCH/InverseSr-Astro/project/BRGM_ddim.py \    
    --update_latent_variables \
    --update_conditioning \




