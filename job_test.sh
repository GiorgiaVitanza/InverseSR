#!/bin/bash

#SBATCH --job-name=TestSR_3D
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --account=IscrC_DATIV-ML
#SBATCH --output=test_vae%j.out
#SBATCH --error=test_vae%j.err
#SBATCH --mem=160G

# Variabili utili per Python
export PYTHONUNBUFFERED=1

# ==============================================================================
# SETUP AMBIENTE E MODULI
# ==============================================================================
module purge
# Load necessary modules (adjust to your environment)
module load cuda/12.2          
module load python/3.11.7

source .venv/bin/activate




python project/test_vae.py   --test_dir "./data/inputs/128x128x128_stride128/npy_patches_test" --data_dir "./data/inputs/128x128x128_stride128/npy_patches" --output_dir "./data/outputs/test_vae_10_z16" --catalogue_path "./data/inputs/128x128x128_stride128/test_catalog.csv" --vae_path "./mlruns/1/models/m-00ba209ea30a49c991d4f83cf30a94f2/artifacts/data/model.pth"
