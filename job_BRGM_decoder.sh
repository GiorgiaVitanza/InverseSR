#!/bin/bash

#SBATCH --job-name=InverseSR_3D
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --account=IscrC_DATIV-ML
#SBATCH --output=InverseSR_decoder_1ch_None_new%j.out
#SBATCH --error=InverseSR_decoder_1ch_None_new%j.err


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


export PYTORCH_ALLOC_CONF=expandable_segments:True


python3 $SCRATCH/InverseSr-Astro/project/BRGM_decoder.py --inference --image_size 128 128 128 --path_to_latent_ddpm "./data/outputs/BRGM_ddim_cond_cross_attn/None/results.pth" --path_to_ddpm_checkpoint "./data/trained_models_astro/trained_models_ddpm_100/None"  --mean_latent_vector

