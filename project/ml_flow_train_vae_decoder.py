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
# Crea un nome unico basato sull'orario e sui parametri
current_time = datetime.now().strftime('%b%d_%H-%M-%S')



mlflow.set_tracking_uri(f"sqlite:///mlruns_vae_decoder_{train_param.epochs}epochs_{current_time}.db")
mlflow.set_experiment(f"Radio_VAE_Hybrid_Logging_{train_param.epochs}epochs_z{hparams.z_channels}_{current_time}")


def run_step(model, x):
    # --- Encoding & Reparameterization ---
    h = model.encoder(x)
    moments_mu = model.quant_conv_mu(h)
    moments_log_var = model.quant_conv_log_sigma(h)
    
    # Reparameterization trick
    std = torch.exp(0.5 * moments_log_var)
    eps = torch.randn_like(std)
    z = moments_mu + eps * std
    
    # --- Decoding ---
    x_hat = model.decode(z)
    
    # --- Loss Calculation ---
    
    # 1. Reconstruction Loss (L1)
    recon_loss = F.l1_loss(x_hat, x, reduction='mean')
    
    # 2. KL Divergence Loss
    # Formula: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
    kl_loss = -0.5 * torch.sum(1 + moments_log_var - moments_mu.pow(2) - moments_log_var.exp(), dim=[1, 2, 3, 4])
    kl_loss = kl_loss.mean()
    
    # Peso della KL (Beta)
    kl_weight = 1e-6 
    
    total_loss = recon_loss + (kl_weight * kl_loss)
    
    return total_loss, recon_loss, kl_loss, x_hat

def train(): 
    CHECKPOINT_DIR = os.path.join(BASE_SCRATCH, f"checkpoints_vae_decoder_{hparams.z_channels}_{train_param.epochs}epochs")
    # Configurazione Log
    TB_LOG_DIR = train_param.tensor_board_logger_vae
  
    log_dir = f"{TB_LOG_DIR}/run_{current_time}_lr_{train_param.learning_rate}"

    dataset = RadioPatchDataset(data_dir=train_param.data_dir, catalogue_path=train_param.catalogue_path, in_channels=hparams.in_channels)
    dataloader = DataLoader(dataset, batch_size=train_param.batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)

    
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

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

            # --- LOG VISIVO POTENZIATO ---
            if epoch % 5 == 0:
                model.eval()
                with torch.no_grad():
                    # Prendiamo il primo sample del batch
                    img_orig = x[0, 0].cpu().numpy()      # Cubo originale (128, 128, 128)
                    img_recon = x_hat[0, 0].cpu().numpy() # Cubo ricostruito

                    # 1. Calcoliamo la Slice Centrale
                    mid_z = img_orig.shape[0] // 2
                    slice_orig = img_orig[mid_z]
                    slice_recon = img_recon[mid_z]

                    # 2. Calcoliamo il MOMENTO 0 (Somma lungo Z)
                    # Questo fa emergere la galassia anche se è debole
                    mom0_orig = np.sum(img_orig, axis=0)
                    mom0_recon = np.sum(img_recon, axis=0)

                    # Creiamo una griglia 2x2 per il confronto
                    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                    
                    # Riga 1: Slice Centrali
                    im1 = axes[0, 0].imshow(slice_orig, cmap='hot')
                    axes[0, 0].set_title(f"Originale (Slice Z={mid_z})")
                    plt.colorbar(im1, ax=axes[0, 0])
                    
                    im2 = axes[0, 1].imshow(slice_recon, cmap='hot')
                    axes[0, 1].set_title("Ricostruito (Slice)")
                    plt.colorbar(im2, ax=axes[0, 1])

                    # Riga 2: Momento 0 (Proiezioni)
                    # Usiamo vmax basato sul 99° percentile per il contrasto
                    vmax_mom = np.percentile(mom0_orig, 99.9)
                    im3 = axes[1, 0].imshow(mom0_orig, cmap='hot', vmax=vmax_mom)
                    axes[1, 1].set_title("Originale (Momento 0)")
                    plt.colorbar(im3, ax=axes[1, 0])

                    im4 = axes[1, 1].imshow(mom0_recon, cmap='hot', vmax=vmax_mom)
                    axes[1, 1].set_title("Ricostruito (Momento 0)")
                    plt.colorbar(im4, ax=axes[1, 1])

                    # Log su TensorBoard e MLflow
                    writer.add_figure("Visual/3D_Comparison", fig, global_step=epoch)
                    
                    plot_path = f"epoch_{epoch}_recon.png"
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
            registered_model_name=f"VAE_{hparams.in_channels}ch",
            export_model=True
        )
        local_vae_pack = os.path.join(OUTPUT_DIR, "VAE_full")
        mlflow.pytorch.save_model(model, path=local_vae_pack)
        
        # 2. Impacchetta solo il decoder per inferenza
        only_decoder = OnlyDecoder(model)
        mlflow.pytorch.log_model(
            pytorch_model=only_decoder, 
            name="decoder_only_model",
            registered_model_name=f"Decoder_{hparams.in_channels}ch",
            export_model=True
        )
        local_decoder_pack = os.path.join(OUTPUT_DIR, "Decoder_only")
        mlflow.pytorch.save_model(only_decoder, path=local_decoder_pack)

    writer.close()
    print(f"Training concluso. Checkpoint fisici in {CHECKPOINT_DIR}, log in {TB_LOG_DIR}")

if __name__ == "__main__":
    train()