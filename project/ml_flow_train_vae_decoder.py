import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import mlflow
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
# --- TENSORBOARD ---
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# Import dei tuoi moduli
from utils.dataset_v3 import RadioPatchDataset 
from models.aekl_no_attention import AutoencoderKL, OnlyDecoder
from utils.config_aekl_v3 import get_hparams 
from utils.config_train import train_config

# --- CONFIGURAZIONE AMBIENTE LEONARDO ---
hparams, unknown = get_hparams()
train_param, _ = train_config()
BASE_SCRATCH = f"/leonardo_scratch/large/userexternal/gvitanza/InverseSR/"
OUTPUT_DIR = train_param.output_dir_vae
CHECKPOINT_DIR = os.path.join(BASE_SCRATCH, f"checkpoints_vae_decoder_{hparams.in_channels}_{train_param.epochs}epochs")
# Configurazione Log
TB_LOG_DIR = train_param.tensor_board_logger_vae
# Crea un nome unico basato sull'orario e sui parametri
current_time = datetime.now().strftime('%b%d_%H-%M-%S')
log_dir = f"{TB_LOG_DIR}/run_{current_time}_lr_{train_param.learning_rate}"





mlflow.set_tracking_uri(f"sqlite:///mlruns_vae_decoder_{train_param.epochs}epochs.db")
mlflow.set_experiment("Radio_VAE_Hybrid_Logging")

def run_step(model, x):
    h = model.encoder(x)
    moments_mu = model.quant_conv_mu(h)
    moments_log_var = model.quant_conv_log_sigma(h)
    
    std = torch.exp(0.5 * moments_log_var)
    eps = torch.randn_like(std)
    z = moments_mu + eps * std
    
    x_hat = model.decode(z)
    
    recon_loss = F.mse_loss(x_hat, x)
    kl_loss = -0.5 * torch.sum(1 + moments_log_var - moments_mu.pow(2) - moments_log_var.exp())
    kl_loss = kl_loss.mean() * 1e-6 
    
    return recon_loss + kl_loss, recon_loss, kl_loss, x_hat

def train(): 
    dataset = RadioPatchDataset(data_dir=train_param.data_dir, catalogue_path=train_param.catalogue_path, in_channels=hparams.in_channels)
    dataloader = DataLoader(dataset, batch_size=train_param.batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)


    
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(TB_LOG_DIR, exist_ok=True)

    hparams_dict = vars(hparams)
    model = AutoencoderKL(embed_dim=hparams.z_channels, hparams=hparams_dict).to(train_param.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_param.learning_rate)

    # Inizializzazione Loggers
    writer = SummaryWriter(log_dir=log_dir)

    with mlflow.start_run(run_name=f"VAE_Hybrid_Training"):
        mlflow.log_params(hparams_dict)

        for epoch in range(train_param.epochs):
            model.train()
            epoch_total_loss, epoch_recon_loss, epoch_kl_loss = [], [], []
            pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

            for batch in pbar:
                x = batch["x_0"].to(train_param.device)
                optimizer.zero_grad()
                total_loss, rec_loss, kl_loss, x_hat = run_step(model, x)
                total_loss.backward()
                optimizer.step()
                
                epoch_total_loss.append(total_loss.item())
                epoch_recon_loss.append(rec_loss.item())
                epoch_kl_loss.append(kl_loss.item())
                pbar.set_postfix({"total": f"{total_loss.item():.4f}"})

            # --- LOGGING (TensorBoard) ---
            avg_total = np.mean(epoch_total_loss)
            writer.add_scalar("Loss/Total", avg_total, epoch)
            writer.add_scalar("Loss/Recon", np.mean(epoch_recon_loss), epoch)
            writer.add_scalar("Loss/KL", np.mean(epoch_kl_loss), epoch)
            
            # --- LOGGING (MLflow Metrics) ---
            mlflow.log_metric("avg_total_loss", avg_total, step=epoch)

            # --- LOG VISIVO (Entrambi) ---
            if epoch % 5 == 0:
                model.eval()
                with torch.no_grad():
                    mid_idx = x.shape[2] // 2
                    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                    axes[0].imshow(x[0, 0, mid_idx].cpu(), cmap='hot')
                    axes[0].set_title("Originale")
                    axes[1].imshow(x_hat[0, 0, mid_idx].cpu(), cmap='hot')
                    axes[1].set_title("Ricostruito")
                    
                    # Log su TensorBoard
                    writer.add_figure("Visual/Comparison", fig, global_step=epoch)
                    
                    # Log su MLflow come artefatto (opzionale)
                    plot_path = f"tmp_recon.png"
                    plt.savefig(plot_path)
                    mlflow.log_artifact(plot_path, artifact_path="plots")
                    plt.close(fig)
                    if os.path.exists(plot_path): os.remove(plot_path)
                model.train()

            # --- SALVATAGGIO CHECKPOINTS FISICI ---
            if (epoch + 1) % 10 == 0 or (epoch + 1) == train_param.epochs:
                vae_path = os.path.join(CHECKPOINT_DIR, f"vae_full_ep{epoch+1}.pth")
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'hparams': hparams_dict}, vae_path)

        # --- SALVATAGGIO FINALE MODELLO (IMPACCHETTAMENTO MLFLOW) ---
        print("Registrazione modelli su MLflow...")
        
        # 1. Impacchetta il modello completo
        mlflow.pytorch.log_model(
            pytorch_model=model, 
            name="vae_full_model",
            registered_model_name=f"VAE_{hparams.in_channels}ch"
        )
        local_vae_pack = os.path.join(OUTPUT_DIR, "VAE_full")
        mlflow.pytorch.save_model(model, path=local_vae_pack)
        
        # 2. Impacchetta solo il decoder per inferenza
        only_decoder = OnlyDecoder(model)
        mlflow.pytorch.log_model(
            pytorch_model=only_decoder, 
            name="decoder_only_model",
            registered_model_name=f"Decoder_{hparams.in_channels}ch"
        )
        local_decoder_pack = os.path.join(OUTPUT_DIR, "Decoder_only")
        mlflow.pytorch.save_model(only_decoder, path=local_decoder_pack)

    writer.close()
    print(f"Training concluso. Checkpoint fisici in {CHECKPOINT_DIR}, log in {TB_LOG_DIR}")

if __name__ == "__main__":
    train()