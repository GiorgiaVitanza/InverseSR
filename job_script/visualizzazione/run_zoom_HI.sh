#!/bin/bash
#SBATCH --job-name=HI_GPU_Viz
#SBATCH --account=IscrC_DATIV-ML       
#SBATCH --partition=boost_usr_prod       # Partizione GPU Booster di Leonardo
#SBATCH --nodes=1                        # 1 Nodo
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G                 # Richiede 64 GB di RAM CPU (oppure 128G se il cubo è gigante)
#SBATCH --gres=gpu:1                   # Richiede 1 GPU NVIDIA A100
#SBATCH --time=00:30:00                  # Tempo massimo (30 minuti)
#SBATCH --output=hi_viz_%j.out
#SBATCH --error=hi_viz_%j.err

# Pulizia ambiente e caricamento dei moduli CINECA con PyTorch abilitato su GPU

SCRATCH=/leonardo_scratch/large/userexternal/gvitanza/InverseSR
HOME=/leonardo/home/userexternal/gvitanza/

source ${HOME}/.venv/bin/activate
# Verifica che la GPU sia visibile a PyTorch prima dell'esecuzione
python3 -c "import torch; print('🚀 CUDA disponibile:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Nessuna')"

# Esecuzione del tuo script Python per i dati HI
python3 ${SCRATCH}/project/zoom_HI.py