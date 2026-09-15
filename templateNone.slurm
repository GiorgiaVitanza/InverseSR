#!/bin/bash
#SBATCH --job-name=Inv_ddim_{EXP_NAME}
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --account=IscrC_DATIV-ML
#SBATCH --output=logs/out___EXP_NAME___%j.out
#SBATCH --error=logs/err___EXP_NAME___%j.err

module purge
module load cuda/12.2
module load python/3.11.7

SCRATCH=/leonardo_scratch/large/userexternal/gvitanza/InverseSR
HOME=/leonardo/home/userexternal/gvitanza/

source ${HOME}/.venv/bin/activate

# Parametri sintonizzati
LEARNING_RATE=__LR__
LAMBDA_PERC=__LAMBDA__
NUM_STEPS=__STEPS__
EXPERIMENT_NAME="__EXP_NAME__"

LOG_DIR=$SCRATCH/generated_jobs/$EXPERIMENT_NAME

VAE="${SCRATCH}/data/trained_models_astro/vae_decoder_300_z3_global_robust_cont_ldev_augment_5/VAE_full"
DDPM="${SCRATCH}/data/trained_models_astro/ddpm_None_300_z3_global_robust_cont_ldev_logvarFalse_augment_5/ddpm_final_model"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.8"


python3 ${SCRATCH}/project/BRGM_ddim_cond_v2.py \
    --input_data_dir "${SCRATCH}/data/inputs/16x128x128_stride128_cont_ldev/Inversion"\
    --image_size 16 128 128 \
    --inference \
    --z_channels 3 \
    --cond_key "None"\
    --out_channels 1\
    --ddim_num_timesteps 70\
    --num_steps $NUM_STEPS\
    --ddim_eta 0.0 \
    --update_latent_variables \
    --mean_latent_vector \
    --prior_every 15 \
    --data_format="npy" \
    --corruption="downsample" \
    --lambda_perc="$LAMBDA_PERC" \
    --learning_rate=$LEARNING_RATE \
    --experiment_name=$EXPERIMENT_NAME \
    --downsample_factor 4 \
    --tensor_board_logger_ddim="$LOG_DIR" \
    --output_dir_BRGM_ddim="${SCRATCH}/data/outputs/BRGM_ddim_${EXPERIMENT_NAME}" \
    --norm_data "global_robust" \
    --vae_path_BRGM $VAE\
    --ddpm_path_BRGM $DDPM \
    