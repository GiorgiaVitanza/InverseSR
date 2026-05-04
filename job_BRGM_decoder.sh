#!/bin/bash

#SBATCH --job-name=InverseSR_3D
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --account=IscrC_DATIV-ML
#SBATCH --output=InverseSR_decoder_z8_lambda1e4%j.out
#SBATCH --error=InverseSR_decoder_z8_lambda1e4%j.err


# Variabili utili per Python
export PYTHONUNBUFFERED=1

# ==============================================================================
# SETUP AMBIENTE E MODULI
# ==============================================================================
module purge
# Load necessary modules (adjust to your environment)
module load cuda/12.2
module load python/3.11.7

source $SCRATCH/InverseSR/.venv/bin/activate
echo -e '\n\n\n'
echo "$(date +"%T"):  start running model!"


export PYTORCH_ALLOC_CONF=expandable_segments:True


python3 $SCRATCH/InverseSR/project/BRGM_decoder.py \
    --inference \
    --image_size 128 128 128 \
    --path_to_latent_ddpm "./data/outputs/BRGM_ddim_cond_z8_down4/results.pth"\
    --path_to_ddpm_checkpoint "./data/trained_models_astro/ddpm_cross_attn_10_2_z8_local/ddpm_final_model"  \
    --norm_data "local" \
    --tensor_board_logger ./logs/BRGM_Decoder_z8_down4 \
	--z_channels 8 \
    --num_step 100 \
    --learning_rate 7e-2 \
    --lambda_perc 1e4 \
    --downsample_factor 4 \
    --corruption downsample \
    --output_dir_BRGM_decoder "./data/outputs/BRGM_decoder_z8_down4"

