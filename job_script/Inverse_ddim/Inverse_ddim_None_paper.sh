#!/bin/bash

#SBATCH --job-name=Inv_ddim
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --account=IscrC_DATIV-ML
#SBATCH --output=Inv_ddim_robust_%j.out
#SBATCH --error=Inv_ddim_robust_%j.err





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
VAE="${SCRATCH}/data/trained_model_paper/vae_decoder_3_300epochs_global_robust_Oct10_20-55-53_best_val/VAE_full_best"
DDPM="${SCRATCH}/data/trained_model_paper/ddpm_None_1000_z3_global_robust_16x128x128_cont_ldev_OK_augm_vae_best/ddpm_best_model"
NORM_DATA='global_robust'
LAMBDA_PRIOR=0
LEARNING_RATE=0.1
LAMBDA_PERC=0
CORRUPTION=downsample
PRIOR_EVERY=1
DATA_FORMAT="npy"
DOWNSAMPLE_FACTOR=4
DDIM_ETA=0.0
COND_KEY="None"
NUM_STEPS=1000
DDIM_NUM_STEPS=70
EXPERIMENT_NAME=z3_lambda${LAMBDA_PERC}_${NORM_DATA}_${DATASET}_${COND_KEY}_${NUM_STEPS}_lr0.1_lp0_pe1_0000_pat20
Z_CHANNELS=3
LOG_DIR=$SCRATCH/logs_paper/$EXPERIMENT_NAME


export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.8"

python3 ${SCRATCH}/project/BRGM_ddim_cond_v2.py \
    --hr_path ${SCRATCH}/data/inputs/${DATASET}/test_augmented/patch_000000_flip_d.npy\
    --lr_path ${SCRATCH}/data/inputs/${DATASET}/test_augmented_LR/patch_000000_flip_d.npy\
    --patience 20 \
    --min_delta 0.0001 \
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
    --output_dir_BRGM_ddim="${SCRATCH}/data/outputs_paper/BRGM_ddim_${EXPERIMENT_NAME}" \
    --norm_data $NORM_DATA \
    --vae_path_BRGM $VAE\
    --ddpm_path_BRGM $DDPM \
    
    

