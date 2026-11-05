#!/bin/bash

#SBATCH --job-name=ddpm_None
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --account=IscrC_DATIV-ML
#SBATCH --output=ddpm_None_HI_global1_vae_best_%j.out
#SBATCH --error=ddpm_None_HI_global1_vae_best_%j.err


module purge

module load cuda/12.2          
module load python/3.11.7

SCRATCH=/leonardo_scratch/large/userexternal/gvitanza/InverseSR
HOME=/leonardo/home/userexternal/gvitanza/

source ${HOME}/.venv/bin/activate


DATASET=HI_datamix_128x128x128_stride64
EPOCHS=50
Z_CHANNELS=3
NORM_MODE='global_1'
COND='None'
EXPERIMENT_NAME=${COND}_${EPOCHS}_z${Z_CHANNELS}_${NORM_MODE}_${DATASET}_vae_best
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 ${SCRATCH}/project/ml_flow_train_ddpm_v3_no_augm.py\
    --data_dir "${SCRATCH}/data/inputs/${DATASET}"\
    --in_channels_unet $Z_CHANNELS\
    --out_channels_unet $Z_CHANNELS\
    --tensor_board_logger_ddpm "${SCRATCH}/logs_ddpm_paper/ddpm_${EXPERIMENT_NAME}"\
    --output_dir_ddpm "${SCRATCH}/data/trained_model_paper/ddpm_${EXPERIMENT_NAME}"\
    --epochs $EPOCHS\
    --z_channels $Z_CHANNELS\
    --norm_mode $NORM_MODE\
    --learning_rate 1e-4\
    --cond_key $COND\
    --vae_path "${SCRATCH}/vae_decoder_3_300epochs_global_1_Oct30_16-14-13/vae_best_model.pth"\
    --no-use_mask_channel \
    --no-use_spatial_transformer \
    --batch_size 1\
