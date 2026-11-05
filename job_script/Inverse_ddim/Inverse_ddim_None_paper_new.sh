#!/bin/bash

#SBATCH --job-name=Inv_ddim
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=15:00:00
#SBATCH --account=IscrC_DATIV-ML
#SBATCH --output=Inv_ddim_global1_new_crop_opt_OK_%j.out
#SBATCH --error=Inv_ddim_global1_new_crop_opt_OK_%j.err

module purge
module load cuda/12.2          
module load python/3.11.7

SCRATCH=/leonardo_scratch/large/userexternal/gvitanza/InverseSR
HOME=/leonardo/home/userexternal/gvitanza/

source ${HOME}/.venv/bin/activate

# run script
echo -e '\n\n\n'
echo "$(date +"%T"): start running model!"

DATASET=16x128x128_cont_ldev_OK
VAE_G1="${SCRATCH}/data/trained_model_paper/vae_decoder_300_z3_global_1_16x128x128_cont_ldev_OK_val_augm/vae_best_model"
DDPM_G1="${SCRATCH}/data/trained_model_paper/ddpm_None_500_z3_global_1_16x128x128_cont_ldev_OK_augm_vae_best/ddpm_best_model"

NORM_DATA='global_1'
LAMBDA_PRIOR=0.08
LEARNING_RATE=0.04
LAMBDA_PERC=0
CORRUPTION=downsample
PRIOR_EVERY=1
DATA_FORMAT="npy"             
DOWNSAMPLE_FACTOR=4
DDIM_ETA=0.0
COND_KEY="None"
NUM_STEPS=500
# ==============================================================================
# CONFIGURAZIONE DEL CROP DAL CUBO GRANDE (Scegli qui la regione di interesse!)
# ==============================================================================
CROP_TOP_Y=256
CROP_LEFT_X=512
CROP_H=256
CROP_W=256
DDIM_NUM_STEPS=45
DELTA=0.01
PATIENCE=10
EXPERIMENT_NAME=CROP${CROP_H}_${CROP_W}_cropcoords${CROP_TOP_Y}_${CROP_LEFT_X}_lambda${LAMBDA_PERC}_${NORM_DATA}_ddimsteps${DDIM_NUM_STEPS}_lr${LEARNING_RATE}_lp${LAMBDA_PRIOR}_pe${PRIOR_EVERY}_pat${PATIENCE}_${DELTA}_opt
Z_CHANNELS=3
LOG_DIR=$SCRATCH/logs_paper/$EXPERIMENT_NAME


python3 ${SCRATCH}/project/BRGM_ddim_cond_crop.py \
    --hr_path ${SCRATCH}/data/inputs/${DATASET}/test_full/cont_ldev.npy \
    --lr_path ${SCRATCH}/data/inputs/${DATASET}/test_full_LR/cont_ldev.npy \
    --crop_top_y $CROP_TOP_Y \
    --crop_left_x $CROP_LEFT_X \
    --crop_h $CROP_H \
    --crop_w $CROP_W \
    --downsample_factor 4 \
    --image_size 16 $CROP_H $CROP_W \
    --data_format "npy" \
    --z_channels 3 \
    --cond_key $COND_KEY \
    --out_channels 1 \
    --batch_size 1\
    --ddim_num_timesteps $DDIM_NUM_STEPS \
    --num_steps $NUM_STEPS \
    --ddim_eta $DDIM_ETA \
    --update_latent_variables \
    --mean_latent_vector \
    --prior_every $PRIOR_EVERY \
    --lambda_prior $LAMBDA_PRIOR \
    --data_format=$DATA_FORMAT \
    --corruption=$CORRUPTION \
    --lambda_perc="$LAMBDA_PERC" \
    --learning_rate=$LEARNING_RATE \
    --experiment_name=$EXPERIMENT_NAME \
    --downsample_factor $DOWNSAMPLE_FACTOR \
    --tensor_board_logger_ddim="$LOG_DIR" \
    --output_dir_BRGM_ddim="${SCRATCH}/data/outputs_paper/BRGM_ddim_${EXPERIMENT_NAME}" \
    --norm_data $NORM_DATA \
    --vae_path_BRGM $VAE_G1 \
    --ddpm_path_BRGM $DDPM_G1