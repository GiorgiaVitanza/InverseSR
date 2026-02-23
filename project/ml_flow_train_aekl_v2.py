import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import mlflow
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import dei tuoi moduli aggiornati
from utils.dataset_v3 import RadioPatchDataset # Usiamo la v3 coerente
from models.aekl_no_attention import AutoencoderKL
from utils.config_aekl_v2 import get_hparams 

# --- CONFIGURAZIONE AMBIENTE LEONARDO ---
BASE_SCRATCH = "/leonardo_scratch/large/userexternal/gvitanza/InverseSr-Astro/data/outputs"
MLFLOW_TRACKING_URI = f"file:{os.path.join(BASE_SCRATCH, 'mlruns_vae')}"
CHECKPOINT_DIR = os.path.join(BASE_SCRATCH, "checkpoints_vae")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# 3. MLFLOW SETUP
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Radio_VAE_v2")

# --- FUNZIONE DI PASSAGGIO (FORWARD) ---
def run_step(model, x):
    """
    Esegue il passaggio completo nel VAE: Encode -> Reparametrize -> Decode.
    """
    # 1. ENCODING
    # h: [B, n_channels*mult, 32, 32, 32] (se image_size=128 e downsampling=4)
    h = model.encoder(x)
    
    # 2. BOTTLENECK (Momenti della distribuzione Gaussiana)
    moments_mu = model.quant_conv_mu(h)
    moments_log_var = model.quant_conv_log_sigma(h)
    
    # 3. REPARAMETRIZATION TRICK
    std = torch.exp(0.5 * moments_log_var)
    eps = torch.randn_like(std)
    z = moments_mu + eps * std
    
    # 4. DECODING
    x_hat = model.decode(z)
    
    # 5. CALCOLO LOSS
    # Reconstruction Loss: Quanto l'output somiglia all'input
    recon_loss = F.mse_loss(x_hat, x)
    
    # KL Divergence: Regolarizza lo spazio latente verso una Gaussiana Standard
    kl_loss = -0.5 * torch.sum(1 + moments_log_var - moments_mu.pow(2) - moments_log_var.exp())
    # Peso KL (beta): Spesso si usa un valore molto basso (1e-6) per i cubi astrofisici
    # per evitare che la KL domini sulla ricostruzione (KL vanishing)
    kl_loss = kl_loss.mean() * 1e-6 
    
    return recon_loss + kl_loss, recon_loss, x_hat

def train():
    # 0. CARICAMENTO HPARAMS
    hparams = get_hparams() # Assicurati che z_channels=3 e resolution=[128,128,128]
    
    # 1. DATASET E DATALOADER (v3)
    dataset = RadioPatchDataset(
        data_dir=hparams.data_dir, 
        catalogue_path=hparams.catalogue_path
    )
    # Batch size basso (1-2) per cubi 3D 128^3 per non saturare la VRAM
    dataloader = DataLoader(dataset, batch_size=hparams.batch_size, shuffle=True, num_workers=2)

    # 2. MODELLO E OTTIMIZZATORE
    hparams_dict = vars(hparams)
    # embed_dim deve corrispondere ai z_channels (solitamente 3)
    model = AutoencoderKL(embed_dim=hparams.z_channels, hparams=hparams_dict).to(hparams.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams.learning_rate)


    with mlflow.start_run(run_name=f"{hparams.experiment_name}_v2"):
        mlflow.log_params(hparams_dict)

        for epoch in range(hparams.epochs):
            model.train()
            pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
            epoch_total_loss = []
            epoch_recon_loss = []

            for batch in pbar:
                # Recuperiamo solo x_0 (il cubo), ignoriamo il context del catalogo nel VAE
                x = batch["x_0"].to(hparams.device)
                
                optimizer.zero_grad()
                
                total_loss, rec_loss, x_hat = run_step(model, x)
                
                total_loss.backward()
                optimizer.step()
                
                epoch_total_loss.append(total_loss.item())
                epoch_recon_loss.append(rec_loss.item())
                
                pbar.set_postfix({
                    "total": f"{total_loss.item():.4f}",
                    "recon": f"{rec_loss.item():.4f}"
                })

            # Log metriche medie
            mlflow.log_metric("avg_total_loss", np.mean(epoch_total_loss), step=epoch)
            mlflow.log_metric("avg_recon_loss", np.mean(epoch_recon_loss), step=epoch)

            # 4. LOG VISIVO (Slice centrale)
            if epoch % 5 == 0:
                model.eval()
                with torch.no_grad():
                    mid_idx = x.shape[2] // 2
                    orig_img = x[0, 0, mid_idx].cpu().numpy()
                    recon_img = x_hat[0, 0, mid_idx].cpu().numpy()

                    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                    im0 = axes[0].imshow(orig_img, cmap='hot')
                    axes[0].set_title("Originale (FITS)")
                    fig.colorbar(im0, ax=axes[0])
                    
                    im1 = axes[1].imshow(recon_img, cmap='hot')
                    axes[1].set_title("Ricostruito (VAE)")
                    fig.colorbar(im1, ax=axes[1])
                    
                    plot_path = f"./data/outputs/vae_recon_epoch_{epoch}.png"
                    plt.savefig(plot_path)
                    mlflow.log_artifact(plot_path)
                    plt.close(fig)
                    if os.path.exists(plot_path): os.remove(plot_path)

            # 5. SALVATAGGIO PERIODICO
            if (epoch + 1) % 10 == 0:
                ckpt_path = os.path.join(CHECKPOINT_DIR, f"vae_ep{epoch+1}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'hparams': hparams_dict
                }, ckpt_path)
            
    # Registrazione ufficiale su MLflow
    mlflow.pytorch.log_model(model, name="VAE_v2")

    print(f"Training concluso. Checkpoints salvati in {CHECKPOINT_DIR}")

if __name__ == "__main__":
    train()