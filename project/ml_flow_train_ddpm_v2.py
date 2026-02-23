import os
import torch
import torch.nn.functional as F
import mlflow
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.aekl_no_attention import AutoencoderKL

# Import dai tuoi moduli
from utils.dataset_v3 import RadioPatchDataset, CatalogueEmbedder
from utils.config_unet_v2 import get_config, train_config
from utils.config_aekl_v2 import get_hparams
from models.ddpm_v2_conditioned import DDPM

# --- CONFIGURAZIONE PERCORSI LEONARDO ---
BASE_SCRATCH = "/leonardo_scratch/large/userexternal/gvitanza/InverseSr-Astro/data/outputs"
MLFLOW_TRACKING_URI = f"file:{os.path.join(BASE_SCRATCH, 'mlruns_ddpm')}"
CHECKPOINT_DIR = os.path.join(BASE_SCRATCH, "checkpoints_ddpm")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# --- MLFLOW SETUP ---
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Radio_DDPM_v2")

# Carica il VAE
hparams = get_hparams() # Assicurati che z_channels=3 e resolution=[128,128,128]
hparams_dict = vars(hparams)
vae = AutoencoderKL(embed_dim=hparams.z_channels, hparams=hparams_dict).to(hparams.device)
#checkpoint = torch.load("outputs from leonardo/checkpoints/vae_1ch_ep100.pth")
checkpoint = torch.load(
    "outputs from leonardo/checkpoints/vae_3ch_ep100.pth", 
    map_location=torch.device('cpu'), # Forza il caricamento su CPU
    weights_only=False
)
vae.load_state_dict(checkpoint['model_state_dict'])
vae.eval() # Importante: il VAE deve essere in modalità eval e non richiede gradienti

def train():
    # Caricamento configurazioni
    unet_cfg = get_config() # Restituisce il dizionario {"params": {...}}
    train_cfg = train_config()
    
    # 1. DATASET E DATALOADER
    dataset = RadioPatchDataset(
        data_dir=train_cfg.data_dir, 
        catalogue_path=train_cfg.catalogue_path
    )
    dataloader = DataLoader(dataset, batch_size=train_cfg.batch_size, shuffle=True, num_workers=4)

    # 2. MODELLI (DDPM + EMBEDDER)
    # Inizializziamo l'embedder (8 parametri -> 128 context_dim)
    embedder = CatalogueEmbedder(input_dim=8, embed_dim=128).to(train_cfg.device)

    model = DDPM(
        unet_config=unet_cfg,
        conditioning_key="crossattn", # Assicurati che sia crossattn per usare l'embedder
        learn_logvar=True
    ).to(train_cfg.device)

    # Ottimizzatore congiunto
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(embedder.parameters()), 
        lr=train_cfg.learning_rate
    )

    # 3. TRAINING LOOP CON MLFLOW
    with mlflow.start_run(run_name=f"{train_cfg.experiment_name}_v2"):
        mlflow.log_params({
            "epochs": train_cfg.epochs,
            "batch_size": train_cfg.batch_size,
            "lr": train_cfg.learning_rate,
            "context_dim": 128,
            "device": str(train_cfg.device)
        })

        for epoch in range(train_cfg.epochs):
            model.train()
            embedder.train()
            pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
            epoch_loss = []

            for batch in pbar:
                optimizer.zero_grad()
                
                # 1. Prendi il cubo originale dal dataset
                x_start = batch["x_0"].to(train_cfg.device) # [B, 1, 128, 128, 128]
                raw_context = batch["context"].to(train_cfg.device)

                # 2. Trasforma il cubo in LATENTE (z) usando il VAE
                with torch.no_grad():
                    # encode() restituisce solitamente una distribuzione o i momenti
                    # z è il vettore latente con 3 canali (es. [B, 3, 32, 32, 32])
                    h = vae.encoder(x_start)
                    mu = vae.quant_conv_mu(h)
                    log_var = vae.quant_conv_log_sigma(h)

                    # Reparameterization trick: z = mu + sigma * epsilon
                    std = torch.exp(0.5 * log_var)
                    eps = torch.randn_like(std)
                    z = mu + eps * std

                # 3. Trasforma il catalogo in embedding
                c_emb = embedder(raw_context)

                # 4. Passa il LATENTE e il CONTEXT alla DDPM
                # Ora la UNet riceverà un input con 3 canali (z) e non avrà più errori!
                loss, loss_dict = model(z, c_emb)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss.append(loss.item())
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            # Log metriche medie
            avg_loss = np.mean(epoch_loss)
            mlflow.log_metric("avg_loss", avg_loss, step=epoch)

            if epoch % 20 == 0:
                model.eval()
                vae.eval() # Assicurati che il VAE sia in eval
                with torch.no_grad():
                    # 1. Ottieni il latente predetto (semplificato)
                    # In DDPM, per un log veloce, possiamo guardare come il VAE 
                    # ricostruisce il latente "pulito" z per verificare che tutto sia collegato bene
                    h = vae.encoder(x_start)
                    mu = vae.quant_conv_mu(h)
                    log_var = vae.quant_conv_log_sigma(h)

                    # Reparameterization trick: z = mu + sigma * epsilon
                    std = torch.exp(0.5 * log_var)
                    eps = torch.randn_like(std)
                    z = mu + eps * std
                    # z = vae.encoder(x_start).sample()
                    x_recon = vae.decode(z) # Torniamo nello spazio fisico [B, 1, 128, 128, 128]

                    mid = x_start.shape[2] // 2
                    orig_slice = x_start[0, 0, mid].cpu().numpy()
                    recon_slice = x_recon[0, 0, mid].cpu().numpy()

                    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                    
                    # Originale
                    im0 = axes[0].imshow(orig_slice, cmap='hot')
                    axes[0].set_title(f"Target Originale (Ep {epoch})")
                    fig.colorbar(im0, ax=axes[0])
                    
                    # Ricostruito dal latente
                    im1 = axes[1].imshow(recon_slice, cmap='hot')
                    axes[1].set_title("Ricostruzione VAE (LDM Space)")
                    fig.colorbar(im1, ax=axes[1])

                    img_path = f"./data/outputs/val_epoch_{epoch}.png"
                    plt.savefig(img_path)
                    mlflow.log_artifact(img_path)
                    plt.close(fig)

            # 5. SALVATAGGIO PERIODICO (Modello + Embedder)
            if (epoch + 1) % 10 == 0:
                ckpt_path = os.path.join(CHECKPOINT_DIR, f"ddpm_astro_ep{epoch+1}.pth")
                
                # Salviamo entrambi gli state_dict in un unico dizionario
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'embedder_state_dict': embedder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                }, ckpt_path)
            
    # Registrazione su MLflow
    mlflow.pytorch.log_model(model, name="DDPM_v2")

    print("Training completato con successo.")

if __name__ == "__main__":
    train()