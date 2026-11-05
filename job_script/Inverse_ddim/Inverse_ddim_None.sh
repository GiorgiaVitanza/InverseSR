#!/bin/bash

#SBATCH --job-name=Inv_ddim
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --account=IscrC_DATIV-ML
#SBATCH --output=Inv_ddim_local_NOaugm_%j.out
#SBATCH --error=Inv_ddim_local_NOaugm_%j.err





module purge

module load cuda/12.2          
module load python/3.11.7

SCRATCH=/leonardo_scratch/large/userexternal/gvitanza/InverseSR
HOME=/leonardo/home/userexternal/gvitanza/

source ${HOME}/.venv/bin/activate

# run script
echo -e '\n\n\n'
echo "$(date +"%T"):  start running model!"

DATASET=16x128x128_cont_ldev_OK
VAE="${SCRATCH}/data/trained_model_proceeding/vae_decoder_3_300epochs_local_no_augm_Oct10_22-29-35_best_val/VAE_full_best"
DDPM="${SCRATCH}/data/trained_model_proceeding/ddpm_None_1000_z3_local_16x128x128_cont_ldev_OK_NOaugm_vae_best/ddpm_best_model"
NORM_DATA='local'
LAMBDA_PRIOR=0.05
LEARNING_RATE=0.1
LAMBDA_PERC=0
CORRUPTION=downsample
PRIOR_EVERY=1
DATA_FORMAT="npy"
DOWNSAMPLE_FACTOR=4
DDIM_ETA=0.0
COND_KEY="None"
NUM_STEPS=1000
DDIM_NUM_STEPS=50
EXPERIMENT_NAME=lambda${LAMBDA_PERC}_${NORM_DATA}_ddim${DDIM_NUM_STEPS}_${COND_KEY}_${NUM_STEPS}_lr${LEARNING_RATE}_lp${LAMBDA_PRIOR}_pe${PRIOR_EVERY}_0009_pat15_0.001
Z_CHANNELS=3
LOG_DIR=$SCRATCH/logs_proceeding/$EXPERIMENT_NAME


export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.8"

python3 ${SCRATCH}/project/BRGM_ddim_cond_v2.py \
    --hr_path ${SCRATCH}/data/inputs/${DATASET}/test/npy_patches/patch_000009.npy\
    --lr_path ${SCRATCH}/data/inputs/${DATASET}/test_LR/patch_000009.npy\
    --patience 15 \
    --min_delta 0.001 \
    --image_size 16 128 128 \
    --inference \
    --z_channels $Z_CHANNELS \
    --cond_key $COND_KEY\
    --out_channels 1\
    --ddim_num_timesteps $DDIM_NUM_STEPS\
    --num_steps $NUM_STEPS\
    --ddim_eta $DDIM_ETA \
    --update_latent_variables \
    --mean_latent_vector \
    --prior_every $PRIOR_EVERY \
    --lambda_prior $LAMBDA_PRIOR\
    --data_format=$DATA_FORMAT \
    --corruption=$CORRUPTION \
    --lambda_perc="$LAMBDA_PERC" \
    --learning_rate=$LEARNING_RATE \
    --experiment_name=$EXPERIMENT_NAME \
    --downsample_factor $DOWNSAMPLE_FACTOR \
    --tensor_board_logger_ddim="$LOG_DIR" \
    --output_dir_BRGM_ddim="${SCRATCH}/data/outputs_proceeding/BRGM_ddim_${EXPERIMENT_NAME}" \
    --norm_data $NORM_DATA \
    --vae_path_BRGM $VAE\
    --ddpm_path_BRGM $DDPM \
    
    

