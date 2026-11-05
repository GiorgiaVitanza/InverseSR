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
LAMBDA_PRIOR=__PRIOR__
PRIOR_EVERY=__PRIOR_EVERY__

LOG_DIR=$SCRATCH/generated_jobs_proceeding/$EXPERIMENT_NAME

DATASET=16x128x128_cont_ldev_OK
VAE="${SCRATCH}/data/trained_model_proceeding/vae_decoder_3_300epochs_local_no_augm_Oct10_22-29-35_best_val/VAE_full_best"
DDPM="${SCRATCH}/data/trained_model_proceeding/ddpm_None_3_1000epochs_local_Oct11_19-32-17_NOaugm/ddpm_final_model"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.8"


python3 ${SCRATCH}/project/BRGM_ddim_cond_v2.py \
    --hr_path ${SCRATCH}/data/inputs/${DATASET}/test/npy_patches/patch_000033.npy\
    --lr_path ${SCRATCH}/data/inputs/${DATASET}/test_LR/patch_000033.npy\
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
    --lambda_prior $LAMBDA_PRIOR\
    --prior_every $PRIOR_EVERY \
    --data_format="npy" \
    --corruption="downsample" \
    --lambda_perc="$LAMBDA_PERC" \
    --learning_rate=$LEARNING_RATE \
    --experiment_name=$EXPERIMENT_NAME \
    --downsample_factor 4 \
    --tensor_board_logger_ddim="$LOG_DIR" \
    --output_dir_BRGM_ddim="${SCRATCH}/data/outputs_proceeding_parameters_choice/${EXPERIMENT_NAME}" \
    --norm_data "local" \
    --vae_path_BRGM $VAE\
    --ddpm_path_BRGM $DDPM \
    