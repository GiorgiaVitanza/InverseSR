#!/bin/bash

#SBATCH --job-name=Inv_decoder
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --account=IscrC_DATIV-ML
#SBATCH --output=Inv_decoder_%j.out
#SBATCH --error=Inv_decoder_%j.err





module purge

module load cuda/12.2          
module load python/3.11.7

SCRATCH=/leonardo_scratch/large/userexternal/gvitanza/InverseSR

HOME=/leonardo/home/userexternal/gvitanza

source ${HOME}/.venv/bin/activate
# run script
echo -e '\n\n\n'
echo "$(date +"%T"):  start running model!"

VAE="${SCRATCH}/data/trained_models_astro/vae_decoder_train_100ep_z3_local_1e-4_newloss/VAE_full"
DDPM="${SCRATCH}/data/trained_models_astro/ddpm_concat_100_2_z7_local/ddpm_final_model"
BRGM_DDIM="${SCRATCH}/data/outputs/BRGM_ddim_z3_lambda1000_concat_local/results.pth"
NORM_DATA='local'
Z_CHANNELS=3
START_STEPS=0
NUM_STEPS=500
LAMBDA_PRIOR=0
LEARNING_RATE=1e-3
LAMBDA_PERC=1000
CORRUPTION=downsample
PRIOR_EVERY=15
DATA_FORMAT="npy"
DOWNSAMPLE_FACTOR=4
DDIM_NUM_TIMESTEPS=50
DDIM_ETA=0.0
EXPERIMENT_NAME=z3_lambda1000_500_concat_local
LOG_DIR=$SCRATCH/logs/$EXPERIMENT_NAME



export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=max_split_size_mb:128

python3 ${SCRATCH}/project/BRGM_decoder.py \
    --path_to_latent_ddpm $BRGM_DDIM  \
    --image_size 128 128 128 \
    --z_channels $Z_CHANNELS \
    --ddim_eta=$DDIM_ETA \
    --ddim_num_timesteps=$DDIM_NUM_TIMESTEPS \
    --update_latent_variables \
    --update_conditioning \
    --mean_latent_vector \
    --update_hi_size \
    --update_line_flux_integral \
    --update_i \
    --update_w20 \
    --prior_every=$PRIOR_EVERY \
    --num_steps=$NUM_STEPS \
    --data_format=$DATA_FORMAT \
    --inference \
    --corruption="$CORRUPTION" \
    --lambda_perc="$LAMBDA_PERC" \
    --learning_rate=$LEARNING_RATE \
    --experiment_name=$EXPERIMENT_NAME \
    --downsample_factor="$DOWNSAMPLE_FACTOR" \
    --tensor_board_logger_decoder="$LOG_DIR" \
    --output_dir_BRGM_decoder="${SCRATCH}/data/outputs/BRGM_decoder_${EXPERIMENT_NAME}" \
    --norm_data=$NORM_DATA \
    --vae_path_BRGM $VAE\
    --ddpm_path_BRGM $DDPM \

