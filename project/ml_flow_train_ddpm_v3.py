import os
import torch
import torch.nn.functional as F
import mlflow
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
# --- AGGIUNTA TENSORBOARD ---
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# Import dai tuoi moduli
from models.aekl_no_attention import AutoencoderKL
from utils.dataset_v3 import RadioPatchDataset
from utils.config_unet_v3 import get_config
from utils.config_train import train_config
from utils.config_aekl_v3 import get_hparams
from models.ddpm_v2_conditioned import DDPM



# --- CONFIGURAZIONE PERCORSI E DIRECTORY ---
train_cfg, _ = train_config()
hparams, _ = get_hparams()

# Cartella base per questa run
BASE_SCRATCH = "/leonardo_scratch/large/userexternal/gvitanza/InverseSR/"
RUN_DIR = train_cfg.output_dir_ddpm
# Crea un nome unico basato sull'orario e sui parametri
current_time = datetime.now().strftime('%b%d_%H-%M-%S')

print("Inizio configurazione MLFlow...")
# --- MLFLOW SETUP LOCALE ---
db_path = f"mlruns_ddpm.db"
mlflow.set_tracking_uri(f"sqlite:///{db_path}")
mlflow.set_experiment(f"Radio_DDPM_v2_{train_cfg.epochs}epochs_z{hparams.z_channels}_{current_time}")

print("Caricamento VAE pre-addestrato...")
# --- CARICAMENTO VAE (Pre-trained) ---
vae = AutoencoderKL(embed_dim=hparams.z_channels, hparams=vars(hparams)).to(train_cfg.device)
checkpoint_vae = torch.load(train_cfg.vae_path, map_location=train_cfg.device, weights_only=False)
vae.load_state_dict(checkpoint_vae['model_state_dict'])
vae.eval() 




def train():

    CHECKPOINT_DIR = os.path.join(BASE_SCRATCH, f"checkpoints_ddpm_{hparams.z_channels}_{train_cfg.epochs}epochs")
    TB_LOG_DIR = train_cfg.tensor_board_logger_ddpm
  
    log_dir = f"{TB_LOG_DIR}/run_{current_time}_lr_{train_cfg.learning_rate}_z{hparams.z_channels}"


    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Configurazione UNet
    unet_cfg, _ = get_config()
    unet_cfg["params"]["in_channels"] = unet_cfg["params"]["in_channels_unet"]
    unet_cfg["params"]["out_channels"] = unet_cfg["params"]["out_channels_unet"]
    unet_cfg["params"].pop("out_channels_unet", None)  # Rimuoviamo i parametri specifici del config per evitare confusione
    unet_cfg["params"].pop("in_channels_unet", None)

    # Dataset e DataLoader
    dataset = RadioPatchDataset(
        data_dir=train_cfg.data_dir, 
        catalogue_path=train_cfg.catalogue_path,
        in_channels=hparams.in_channels,
        norm_mode=train_cfg.norm_mode
    )
    dataloader = DataLoader(dataset, batch_size=train_cfg.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    # Modello DDPM
    model = DDPM(
        unet_config=unet_cfg,
        conditioning_key=train_cfg.cond_key, 
        learn_logvar=True
    ).to(train_cfg.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.learning_rate)
    
    # Inizializzazione Loggers
    writer = SummaryWriter(log_dir=log_dir)

    with mlflow.start_run(run_name=f"DDPM_Training_{current_time}"):
        mlflow.log_params(vars(train_cfg))
        mlflow.log_params({f"vae_{k}": v for k, v in vars(hparams).items()})
        mlflow.log_params({f"unet_{k}": v for k, v in unet_cfg["params"].items()})
       

        for epoch in range(train_cfg.epochs):
            model.train()
            pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
            epoch_loss = []

            for batch in pbar:
                optimizer.zero_grad()
                
                x_start = batch["x_0"].to(train_cfg.device)
                raw_context = batch["context"].to(train_cfg.device)

                # 1. Encoding nel Latent Space (z)
                with torch.no_grad():
                    h = vae.encoder(x_start)
                    mu = vae.quant_conv_mu(h)
                    log_var = vae.quant_conv_log_sigma(h)
                    std = torch.exp(0.5 * log_var)
                    eps = torch.randn_like(std)
                    z = mu + eps * std

                # 2. Forward DDPM (Diffusion Loss)
                loss, loss_dict = model(z, raw_context)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss.append(loss.item())
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            # --- LOGGING METRICHE (TensorBoard & MLflow) ---
            avg_loss = np.mean(epoch_loss)
            writer.add_scalar("Loss/Train_DDPM", avg_loss, epoch)
            mlflow.log_metric("avg_loss", avg_loss, step=epoch)

            """ if epoch % 5 == 0:
                model.eval()
                with torch.no_grad():
                    # 1. PREPARAZIONE CONDIZIONAMENTO
                    # Prendiamo i primi 2 campioni del contesto dal batch corrente
                    # Il DDPM condizionato ha bisogno di sapere 'cosa' generare
                    curr_cond = raw_context[:2] 

                    # 2. GENERAZIONE DAL DDPM
                    # Ora passiamo esplicitamente il condizionamento al metodo sample
                    # Nota: batch_size=2 perché stiamo usando raw_context[:2]
                    z_gen = model.sample(conditioning=curr_cond, batch_size=2) 
                    
                    # 3. DECODIFICA (Latent -> Image Space)
                    # z_gen è nello spazio dei latenti del VAE
                    x_gen = vae.decode(z_gen)
                    
                    # 4. VISUALIZZAZIONE
                    # Prendiamo il primo canale del primo elemento del batch
                    # Assicurati che x_start e x_gen siano [B, C, D, H, W]
                    img_orig = x_start[0, 0].detach().cpu().numpy()
                    img_gen = x_gen[0, 0].detach().cpu().numpy()
                    
                    mid = img_orig.shape[0] // 2 # Slice centrale sulla profondità (D)
                    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                    
                    # Riga 1: Slice Centrali (D-plane)
                    axes[0, 0].imshow(img_orig[mid], cmap='hot')
                    axes[0, 0].set_title("Originale (Target)")
                    
                    axes[0, 1].imshow(img_gen[mid], cmap='hot')
                    axes[0, 1].set_title(f"DDPM Generated (Epoca {epoch})")

                    # Riga 2: Momento 0 (Proiezioni/Somma lungo l'asse D)
                    # Utile per vedere la struttura radio totale
                    proj_orig = np.sum(img_orig, axis=0)
                    proj_gen = np.sum(img_gen, axis=0)
                    
                    vmax = np.percentile(proj_orig, 99.9)
                    
                    axes[1, 0].imshow(proj_orig, cmap='hot', vmax=vmax)
                    axes[1, 0].set_title("Momento 0 Originale")
                    
                    axes[1, 1].imshow(proj_gen, cmap='hot', vmax=vmax)
                    axes[1, 1].set_title("Momento 0 Generato")

                    # Logging
                    writer.add_figure("Visual/DDPM_Sample", fig, global_step=epoch)
                    # Opzionale: logga anche su MLflow se vuoi vederlo nella UI
                    mlflow.log_figure(fig, f"samples/epoch_{epoch}.png")
                    
                    plt.close(fig)

                model.train() """
            # --- SALVATAGGIO CHECKPOINT PERIODICO ---
            if (epoch + 1) % 20 == 0 or (epoch + 1) == train_cfg.epochs:
                ckpt_path = os.path.join(CHECKPOINT_DIR, f"ddpm_ep{epoch+1}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, ckpt_path)

        # --- SALVATAGGIO FINALE MODELLO (IMPACCHETTAMENTO MLFLOW) ---
        print("Registrazione modello DDPM finale...")
        
        
        # 1. Log interno al DB MLflow
        mlflow.pytorch.log_model(
            pytorch_model=model, 
            name="ddpm",
            registered_model_name=f"DDPM_{hparams.z_channels}ch"
        )
        
        # 2. Salvataggio copia fisica locale in RUN_DIR
        local_model_path = os.path.join(RUN_DIR, "ddpm_final_model")
        mlflow.pytorch.save_model(model, path=local_model_path)

    writer.close()
    print(f"Training concluso. Checkpoint fisici in {CHECKPOINT_DIR}, log in {log_dir}")

if __name__ == "__main__":
    train()