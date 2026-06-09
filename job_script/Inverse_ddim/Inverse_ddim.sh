#!/bin/bash

#SBATCH --job-name=Inv_ddim
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --account=IscrC_DATIV-ML
#SBATCH --output=Inv_ddim_%j.out
#SBATCH --error=Inv_ddim_%j.err





module purge

module load cuda/12.2          
module load python/3.11.7

SCRATCH=/leonardo_scratch/large/userexternal/gvitanza/InverseSR

source ${SCRATCH}/.venv/bin/activate

# run script
echo -e '\n\n\n'
echo "$(date +"%T"):  start running model!"

VAE="${SCRATCH}/data/trained_models_astro/vae_decoder_train_100ep_z3_local_1e-4_newloss/VAE_full"
DDPM="${SCRATCH}/data/trained_models_astro/ddpm_concat_100_2_z7_local/ddpm_final_model"
NORM_DATA='local'
LAMBDA_PRIOR=0
LEARNING_RATE=7e-2
LAMBDA_PERC=1000
CORRUPTION=downsample
PRIOR_EVERY=15
DATA_FORMAT="npy"
DOWNSAMPLE_FACTOR=4
DDIM_ETA=0.0
EXPERIMENT_NAME=z3_lambda1000_concat_local
Z_CHANNELS=3
LOG_DIR=$SCRATCH/logs/$EXPERIMENT_NAME


export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.8"

python3 ${SCRATCH}/project/BRGM_ddim_cond_v2.py \
    --image_size 128 128 128 \
    --z_channels $Z_CHANNELS \
    --ddim_num_timesteps 50\
    --num_steps 50\
    --ddim_eta=$DDIM_ETA \
    --update_latent_variables \
    --update_conditioning \
    --mean_latent_vector \
    --update_hi_size \
    --update_line_flux_integral \
    --update_i \
    --update_w20 \
    --prior_every=$PRIOR_EVERY \
    --data_format=$DATA_FORMAT \
    --test_mode \
    --corruption="$CORRUPTION" \
    --lambda_perc="$LAMBDA_PERC" \
    --learning_rate=$LEARNING_RATE \
    --experiment_name=$EXPERIMENT_NAME \
    --downsample_factor="$DOWNSAMPLE_FACTOR" \
    --tensor_board_logger_ddim="$LOG_DIR" \
    --output_dir_BRGM_ddim="${SCRATCH}/data/outputs/BRGM_ddim_${EXPERIMENT_NAME}" \
    --norm_data=$NORM_DATA \
    --vae_path_BRGM $VAE\
    --ddpm_path_BRGM $DDPM \

