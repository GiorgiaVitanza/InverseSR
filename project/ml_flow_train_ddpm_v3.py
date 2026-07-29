import os
import torch
import torch.nn.functional as F
import mlflow
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# Import dai tuoi moduli
from models.aekl_no_attention import AutoencoderKL
from utils.dataset_v3 import RadioPatchDataset
from utils.config_unet_v3 import get_config
from utils.config_train import train_config
from utils.config_aekl_v3 import get_hparams

from models.ddpm_v2_conditioned import DDPM
from models.ddim import DDIMSampler
from utils.plot_new import comparison_plots_ok, denormalize_data
from utils.const import IMAGE_SHAPE


# --- CONFIGURAZIONE PERCORSI E DIRECTORY ---
train_cfg, _ = train_config()
hparams, _ = get_hparams()

SCALE_FACTOR_VAE = getattr(train_cfg, 'scale_factor_vae', 0.18215) # Scaling di default o dal tuo config

BASE_SCRATCH = "/leonardo_scratch/large/userexternal/gvitanza/InverseSR/"
RUN_DIR = train_cfg.output_dir_ddpm
current_time = datetime.now().strftime('%b%d_%H-%M-%S')

print("Inizio configurazione MLFlow...")
db_path = f"mlruns_ddpm.db"
mlflow.set_tracking_uri(f"sqlite:///{db_path}")
mlflow.set_experiment(f"Radio_DDPM_v2_{train_cfg.epochs}epochs_z{hparams.z_channels}_{current_time}")

print(f"Caricamento VAE pre-addestrato da {train_cfg.vae_path}")
vae = AutoencoderKL(embed_dim=hparams.z_channels, hparams=vars(hparams)).to(train_cfg.device)
checkpoint_vae = torch.load(train_cfg.vae_path, map_location=train_cfg.device, weights_only=False)
vae.load_state_dict(checkpoint_vae['model_state_dict'])
vae.eval() 


def train():
    CHECKPOINT_DIR = os.path.join(BASE_SCRATCH, f"ddpm_{train_cfg.cond_key}_{hparams.z_channels}_{train_cfg.epochs}epochs_{train_cfg.norm_mode}_{current_time}")
    TB_LOG_DIR = train_cfg.tensor_board_logger_ddpm
    log_dir = f"{TB_LOG_DIR}/run_{current_time}_lr_{train_cfg.learning_rate}_z{hparams.z_channels}"

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 1. Configurazione UNet
    unet_cfg, _ = get_config()
    
    # Se concateni la maschera di posizione al latente z, la UNet deve accettare (z_channels + 1)
    # Se usi solo condizionamento vettoriale, in_channels_unet = z_channels
    use_mask_channel = unet_cfg["params"]["use_mask_channel"]
    in_ch = hparams.z_channels + (1 if use_mask_channel else 0)
    
    unet_cfg["params"]["in_channels"] = in_ch
    unet_cfg["params"]["out_channels"] = hparams.z_channels # Output ricostruisce solo z
    unet_cfg["params"].pop("out_channels_unet", None)
    unet_cfg["params"].pop("in_channels_unet", None)

    # 2. Dataset e DataLoader
    dataset = RadioPatchDataset(
        data_dir=train_cfg.data_dir, 
        catalogue_path=train_cfg.catalogue_path,
        in_channels=hparams.in_channels,
        norm_mode=train_cfg.norm_mode
    )
    dataloader = DataLoader(dataset, batch_size=train_cfg.batch_size, shuffle=True, num_workers=1, pin_memory=True)

    # 3. Modello DDPM Latente
    model = DDPM(
        unet_config=unet_cfg,
        conditioning_key=train_cfg.cond_key, 
        learn_logvar=True
    ).to(train_cfg.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.learning_rate)
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
                
                x_start = batch["x_0"].to(train_cfg.device)          # Shape: (B, 1, D, H, W)
                spatial_mask = batch["spatial_mask"].to(train_cfg.device) # Shape: (B, 1, D, H, W)
                context = batch["context"].to(train_cfg.device)      # Shape: (B, 4)

                # --- STEP 1: ENCODING LATENTE VIA VAE ---
                with torch.no_grad():
                    h = vae.encoder(x_start)
                    z = vae.quant_conv_mu(h)
                    z = z * SCALE_FACTOR_VAE # Scaliamo il latente per il DDPM
                

                # --- STEP 2: PREPARAZIONE CONDIZIONAMENTO MISTO / HYBRID ---
                cond_payload = {}

                # 1. Preparazione canale di concatenazione (Maschera)
                if train_cfg.cond_key in ["concat", "hybrid"]:
                    latent_shape = z.shape[2:]  # (D_lat, H_lat, W_lat)
                    mask_latent = F.interpolate(spatial_mask, size=latent_shape, mode='trilinear', align_corners=False)
                    cond_payload["c_concat"] = [mask_latent]  # Il DDPM concatenerà questo [1, 1, D, H, W] a z [1, 3, D, H, W]

                # 2. Preparazione vettori Cross-Attention (Context)
                if train_cfg.cond_key in ["crossattn", "hybrid"] and context is not None:
                    context_attn = context.unsqueeze(1) if context.ndim == 2 else context
                    cond_payload["c_crossattn"] = [context_attn]


                # --- STEP 3: FORWARD & LOSS DDPM ---
                # Il DDPM gestisce internamente la scelta di t e l'aggiunta di rumore su z_input
                loss, loss_dict = model(z, cond_payload)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss.append(loss.item())
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            # Logging metriche
            avg_loss = np.mean(epoch_loss)
            writer.add_scalar("Loss/Train_DDPM", avg_loss, epoch)
            mlflow.log_metric("avg_loss", avg_loss, step=epoch)

            # --- VALIDAZIONE / GENERAZIONE ---
            if epoch % 20 == 0:
                model.eval()
                with torch.no_grad():
                    eval_batch_size = x_start.shape[0]

                    _, _, D_lat, H_lat, W_lat = z.shape
                    shape = (hparams.z_channels, D_lat, H_lat, W_lat)
                    
                    sampler = DDIMSampler(model) 
                    img_noise = torch.randn((eval_batch_size, *shape), device=train_cfg.device)

                    z_gen, _ = sampler.sample(
                        S=50,
                        batch_size=eval_batch_size,
                        shape=shape,
                        conditioning=cond_payload,
                        first_img=img_noise,
                        eta=0.0, 
                        verbose=False
                    ) 
                    
                    z_gen_unscaled = z_gen / SCALE_FACTOR_VAE
                    x_gen = vae.decode(z_gen_unscaled)
                    
                    x_real_denorm = denormalize_data(x_start, train_cfg.norm_mode)
                    x_gen_denorm = denormalize_data(x_gen, train_cfg.norm_mode)
                    
                    # 🎯 ESTRAZIONE COORDINATE DALLA MASCHERA PER IL PRIMO CAMPIONE DEL BATCH (index 0)
                    # Supponendo che spatial_mask[0, 0] sia di shape (D, H, W) con 1 dove c'è la sorgente:
                    mask_sample = spatial_mask[0, 0].cpu().numpy()
                    # Trova gli indici (z, y, x) dove la maschera è attiva (>0.5)
                    src_z, src_y, src_x = np.where(mask_sample > 0.5)
                    coords = list(zip(src_x, src_y, src_z))  # Lista di tuple (x, y, z) per il primo campione
                    
                    # Passiamo le coordinate alla funzione di plot
                    try:
                        fig = comparison_plots_ok(
                            x_real_denorm[0], 
                            x_gen_denorm[0], 
                            sources_coords=coords, 
                            flag='test'
                        )
                        writer.add_figure("Visual/3D_Comparison", fig, global_step=epoch)
                        plt.close(fig)
                    except ValueError as e:
                        print(f"[Warning] Impossibile generare il plot all'epoca {epoch}: {e}")
                        plt.close('all')  # Assicura che la memoria delle figure aperte venga pulita

                model.train()
                            

            # --- CHECKPOINT PERIODICO ---
            if (epoch + 1) % 20 == 0 or (epoch + 1) == train_cfg.epochs:
                ckpt_path = os.path.join(CHECKPOINT_DIR, f"ddpm_ep{epoch+1}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, ckpt_path)

        # --- SALVATAGGIO FINALE ---
        print("Registrazione modello DDPM finale...")
        mlflow.pytorch.log_model(
            pytorch_model=model, 
            artifact_path="ddpm",
            registered_model_name=f"DDPM_{hparams.z_channels}ch"
        )
        
        local_model_path = os.path.join(RUN_DIR, "ddpm_final_model")
        mlflow.pytorch.save_model(model, path=local_model_path)

    writer.close()
    print(f"Training concluso. Checkpoint in {CHECKPOINT_DIR}")

if __name__ == "__main__":
    train()