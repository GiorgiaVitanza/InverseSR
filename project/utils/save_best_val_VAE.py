import os
import torch
import mlflow
import mlflow.pytorch
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
from models.aekl_no_attention import AutoencoderKL, OnlyDecoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_checkpoint_path = "../../vae_decoder_3_300epochs_global_arcsinh_Oct10_20-57-08/vae_best_model.pth"
OUTPUT_DIR = "../../data/trained_model_paper/vae_decoder_3_300epochs_global_arcsinh_Oct10_20-57-08_best_val"

print(f"Caricamento del checkpoint: {best_checkpoint_path}")

# L'argomento map_location sposta i pesi sulla CPU se CUDA non è disponibile
checkpoint = torch.load(best_checkpoint_path, map_location=device)


# 2. Recupera gli iperparametri dal checkpoint (se li hai salvati come hparams)
hparams_dict = checkpoint['hparams']

# 3. Istanzia il modello con gli stessi iperparametri usati nel training

model = AutoencoderKL(embed_dim=hparams_dict['z_channels'], hparams=hparams_dict) 

# 4. Carica i pesi salvati (state_dict) nell'istanza del modello
model.load_state_dict(checkpoint['model_state_dict'])
model.eval() # Imp
# 2. Carica i pesi nell'istanza del modello esistente
model.load_state_dict(checkpoint['model_state_dict'])
model.eval() # Imposta in modalità valutazione

# 3. Impacchettamento e registrazione su MLflow dei pesi della BEST EPOCH
print("Registrazione su MLflow del modello (Best Epoch)...")

# --- VAE Completo ---
mlflow.pytorch.log_model(
    pytorch_model=model, 
    name="vae_full_model_best",
    registered_model_name=f"VAE_best_val"
)
local_vae_pack = os.path.join(OUTPUT_DIR, "VAE_full_best")
# Se la cartella esiste già, rimuovila o cambia nome per evitare conflitti
if os.path.exists(local_vae_pack):
    import shutil
    shutil.rmtree(local_vae_pack)
mlflow.pytorch.save_model(model, path=local_vae_pack)

# --- Solo Decoder ---
only_decoder = OnlyDecoder(model)
mlflow.pytorch.log_model(
    pytorch_model=only_decoder, 
    name="decoder_only_model_best",
    registered_model_name=f"Decoder_best_val"
)
local_decoder_pack = os.path.join(OUTPUT_DIR, "Decoder_only_best")
if os.path.exists(local_decoder_pack):
    import shutil
    shutil.rmtree(local_decoder_pack)
mlflow.pytorch.save_model(only_decoder, path=local_decoder_pack)

print("Modelli della best epoch registrati e salvati correttamente!")