#!/bin/bash

#SBATCH --job-name=InverseSR_3D
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --account=IscrC_DATIV-ML
#SBATCH --output=InverseSR_crossattn_z8_down4_global_sym_new_2_%j.out
#SBATCH --error=InverseSR_crossattn_z8_down4_global_sym_new_2_%j.err


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


python3 $SCRATCH/InverseSR/project/BRGM_ddim_cond_v2.py \
	--update_latent_variables \
	--update_conditioning \
	--update_hi_size \
	--update_i \
	--update_w20 \
	--update_line_flux_integral \
	--inference \
	--image_size 128 128 128\
	--experiment_name "crossattn" \
	--num_steps 50 \
	--ddim_num_timesteps 50\
	--norm_data "local" \
	--tensor_board_logger ./logs/BRGM_ddim_cond_z8_new_1e2_2 \
	--z_channels 8 \
	--downsample_factor 4 \
	--corruption downsample \
	--output_dir_BRGM_ddim "./data/outputs/BRGM_ddim_cond_z8_down4_global_sym_new_1e2_2" \
	--learning_rate 1e-4 \
	--lambda_perc 1e2 \





