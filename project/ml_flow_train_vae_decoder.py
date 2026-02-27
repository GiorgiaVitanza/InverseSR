import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import mlflow
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import dei tuoi moduli
from utils.dataset_v3 import RadioPatchDataset 
from models.aekl_no_attention import AutoencoderKL, OnlyDecoder
from utils.config_aekl_v3 import get_hparams 

# --- CONFIGURAZIONE AMBIENTE LEONARDO ---
BASE_SCRATCH = "/leonardo_scratch/large/userexternal/gvitanza/InverseSr-Astro/data/outputs"
MLFLOW_TRACKING_URI = f"file:{os.path.join(BASE_SCRATCH, 'mlruns_vae')}"
CHECKPOINT_DIR = os.path.join(BASE_SCRATCH, "checkpoints_vae")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Radio_VAE_MultiSave")



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
    
    return recon_loss + kl_loss, recon_loss, x_hat

def train():
    hparams = get_hparams() 
    dataset = RadioPatchDataset(data_dir=hparams.data_dir, catalogue_path=hparams.catalogue_path)
    dataloader = DataLoader(dataset, batch_size=hparams.batch_size, shuffle=True, num_workers=2)

    hparams_dict = vars(hparams)
    model = AutoencoderKL(embed_dim=hparams.z_channels, hparams=hparams_dict).to(hparams.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams.learning_rate)

    with mlflow.start_run(run_name=f"{hparams.experiment_name}_multi_save"):
        mlflow.log_params(hparams_dict)

        for epoch in range(hparams.epochs):
            model.train()
            epoch_total_loss, epoch_recon_loss = [], []
            pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

            for batch in pbar:
                x = batch["x_0"].to(hparams.device)
                optimizer.zero_grad()
                total_loss, rec_loss, x_hat = run_step(model, x)
                total_loss.backward()
                optimizer.step()
                
                epoch_total_loss.append(total_loss.item())
                epoch_recon_loss.append(rec_loss.item())
                pbar.set_postfix({"total": f"{total_loss.item():.4f}"})

            mlflow.log_metric("avg_total_loss", np.mean(epoch_total_loss), step=epoch)
            mlflow.log_metric("avg_recon_loss", np.mean(epoch_recon_loss), step=epoch)

            # --- LOG VISIVO ---
            if epoch % 5 == 0:
                model.eval()
                with torch.no_grad():
                    mid_idx = x.shape[2] // 2
                    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                    axes[0].imshow(x[0, 0, mid_idx].cpu(), cmap='hot')
                    axes[0].set_title("Originale")
                    axes[1].imshow(x_hat[0, 0, mid_idx].cpu(), cmap='hot')
                    axes[1].set_title("Ricostruito")
                    plot_path = f"recon_ep{epoch}.png"
                    plt.savefig(plot_path)
                    mlflow.log_artifact(plot_path)
                    plt.close()
                    if os.path.exists(plot_path): os.remove(plot_path)

            # --- SALVATAGGIO DOPPIO (VAE & DECODER) ---
            if (epoch + 1) % 10 == 0 or (epoch + 1) == hparams.epochs:
                # 1. Checkpoint VAE Completo
                vae_path = os.path.join(CHECKPOINT_DIR, f"vae_full_ep{epoch+1}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'hparams': hparams_dict
                }, vae_path)
                mlflow.log_artifact(vae_path)

                # 2. Checkpoint SOLO DECODER (Pesi puliti per inferenza)
                decoder_path = os.path.join(CHECKPOINT_DIR, f"decoder_only_ep{epoch+1}.pth")
                torch.save({
                    'post_quant_conv': model.post_quant_conv.state_dict(),
                    'decoder': model.decoder.state_dict(),
                    'hparams': hparams_dict
                }, decoder_path)
                mlflow.log_artifact(decoder_path)

        # --- REGISTRAZIONE FINALE SU MLFLOW ---
        # Registra il VAE intero
        mlflow.pytorch.log_model(model,name="model_full_vae")
        
        # Registra solo il Decoder (usando il wrapper)
        only_decoder = OnlyDecoder(model)
        mlflow.pytorch.log_model(only_decoder, name="model_only_decoder")

    print(f"Training concluso. Checkpoints e pesi estratti salvati in {CHECKPOINT_DIR}")

if __name__ == "__main__":
    train()