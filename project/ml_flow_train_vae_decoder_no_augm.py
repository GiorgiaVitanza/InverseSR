import os
from glob import glob
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
from utils.plot_new import comparison_plots_ok, denormalize_data


# --- CONFIGURAZIONE AMBIENTE LEONARDO ---
hparams, unknown = get_hparams()
train_param, _ = train_config()
BASE_SCRATCH = f"/leonardo_scratch/large/userexternal/gvitanza/InverseSR/"
OUTPUT_DIR = train_param.output_dir_vae
# Crea un nome unico basato sull'orario e sui parametri
current_time = datetime.now().strftime('%b%d_%H-%M-%S')

print("Inizializzazione MLflow...")
mlflow.set_tracking_uri(f"sqlite:///mlruns_vae_decoder.db")
mlflow.set_experiment(f"Radio_VAE_Hybrid_Logging_{train_param.epochs}epochs_z{hparams.z_channels}_{current_time}")


def run_step(model, x, epoch=0, total_epochs=train_param.epochs):
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
    
    # 2. Calcolo KL (media per renderla indipendente dalla dimensione del latente)
    kl_loss = -0.5 * torch.mean(1 + moments_log_var - moments_mu.pow(2) - moments_log_var.exp())
    
    # KL Annealing: il peso parte da 0 e arriva a 1e-4 a metà training
    current_kl_weight = min(1e-4, (epoch / (total_epochs / 2)) * 1e-4)
    
    total_loss = recon_loss + (current_kl_weight * kl_loss)
    
    return total_loss, recon_loss, kl_loss, x_hat


def train(): 
    CHECKPOINT_DIR = os.path.join(BASE_SCRATCH, f"vae_decoder_{hparams.z_channels}_{train_param.epochs}epochs_{train_param.norm_mode}_{current_time}")
    # Configurazione Log
    TB_LOG_DIR = train_param.tensor_board_logger_vae
    log_dir = f"{TB_LOG_DIR}/run_{current_time}_lr_{train_param.learning_rate}_z{hparams.z_channels}"
    
    print("Caricamento dataset...")
    dataset = RadioPatchDataset( 
        os.path.join(train_param.data_dir, "train"),
        in_channels=hparams.z_channels, 
        norm_mode=train_param.norm_mode
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=train_param.batch_size, 
        shuffle=True, 
        num_workers=1, 
        pin_memory=True, 
        persistent_workers=True
    )

    val_dataset = RadioPatchDataset(
        data_dir=os.path.join(train_param.data_dir, "val"),
        in_channels=hparams.in_channels,
        norm_mode=train_param.norm_mode,
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=train_param.batch_size,
        shuffle=False,  
        num_workers=1,
        pin_memory=True,
    )

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    hparams_dict = vars(hparams)

    print("Inizializzazione modello, ottimizzatore e scheduler...")
    model = AutoencoderKL(embed_dim=hparams.z_channels, hparams=hparams_dict).to(train_param.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_param.learning_rate)

    # --- INIZIALIZZAZIONE SCHEDULER ---
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=train_param.epochs, 
        eta_min=1e-6
    )

    # Inizializzazione Loggers
    writer = SummaryWriter(log_dir=log_dir)

    # Array per tracciare lo storico delle loss per il grafico finale
    history_train_loss = []
    history_val_loss = []

    best_val_loss = float("inf")
    best_epoch = -1
    with mlflow.start_run(run_name=f"VAE_Hybrid_Training_{current_time}"):
        mlflow.log_params(hparams_dict)
        mlflow.log_params(vars(train_param))

        for epoch in range(train_param.epochs):
            # ================= TRAINING =================
            model.train()
            epoch_total_loss, epoch_recon_loss, epoch_kl_loss = [], [], []
            pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{train_param.epochs}")

            for batch in pbar:
                x = batch["x_0"].to(train_param.device)
                optimizer.zero_grad()
                total_loss, rec_loss, kl_loss, _ = run_step(model, x, epoch=epoch, total_epochs=train_param.epochs)
                total_loss.backward()
                optimizer.step()
                
                epoch_total_loss.append(total_loss.item())
                epoch_recon_loss.append(rec_loss.item())
                epoch_kl_loss.append(kl_loss.item())
                pbar.set_postfix({"train_loss": f"{total_loss.item():.4f}"})

            avg_train_loss = np.mean(epoch_total_loss)
            history_train_loss.append(avg_train_loss)

            # ================= VALIDATION =================
            model.eval()
            val_epoch_losses = []
            with torch.no_grad():
                for val_batch in val_dataloader:
                    x_val = val_batch["x_0"].to(train_param.device)
                    val_tot_loss, _, _, _ = run_step(model, x_val, epoch=epoch, total_epochs=train_param.epochs)
                    val_epoch_losses.append(val_tot_loss.item())

            avg_val_loss = np.mean(val_epoch_losses)
            history_val_loss.append(avg_val_loss)
            # --- SALVATAGGIO CHECKPOINT BEST MODEL IN REAL-TIME ---
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch + 1  # 1-indexed per leggibilità
                best_model_path = os.path.join(CHECKPOINT_DIR, "vae_best_model.pth")
                torch.save({
                    'epoch': best_epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': best_val_loss,
                    'hparams': hparams_dict
                }, best_model_path)
            # --- STEP SCHEDULER & RECUPERO CURRENT LR ---
            current_lr = scheduler.get_last_lr()[0]
            scheduler.step()

            # --- LOGGING (TensorBoard) ---
            writer.add_scalar("Loss/Train_Total", avg_train_loss, epoch)
            writer.add_scalar("Loss/Val_Total", avg_val_loss, epoch)
            writer.add_scalar("Loss/Recon", np.mean(epoch_recon_loss), epoch)
            writer.add_scalar("Loss/KL", np.mean(epoch_kl_loss), epoch)
            writer.add_scalar("Params/LearningRate", current_lr, epoch)
            
           

            # --- LOG VISIVO SU VALIDATION SET (Ogni 20 Epoche) ---
            if epoch % 20 == 0:
                with torch.no_grad():
                    val_batch_sample = next(iter(val_dataloader))
                    x_val_sample = val_batch_sample["x_0"].to(train_param.device)

                    posterior = model.encoder(x_val_sample)
                    z_val = model.quant_conv_mu(posterior)
                    x_hat_val = model.decode(z_val)

                    x_denorm = denormalize_data(x_val_sample, train_param.norm_mode)
                    x_hat_denorm = denormalize_data(x_hat_val, train_param.norm_mode)
                    
                    fig_comp = comparison_plots_ok(x_denorm, x_hat_denorm)
                    writer.add_figure("Visual/3D_Validation_Comparison", fig_comp, global_step=epoch)
                    plt.close(fig_comp)

            # --- SALVATAGGIO CHECKPOINTS FISICI ---
            if (epoch + 1) % 40 == 0 or (epoch + 1) == train_param.epochs:
                vae_path = os.path.join(CHECKPOINT_DIR, f"vae_full_ep{epoch+1}.pth")
                torch.save({
                    'epoch': epoch, 
                    'model_state_dict': model.state_dict(), 
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'hparams': hparams_dict
                }, vae_path)

        # ================= GENERAZIONE E SALVATAGGIO GRAFICO LOSS =================
        print("Generazione grafico Train vs Validation Loss...")
        fig_loss, ax = plt.subplots(figsize=(10, 6))
        epochs_range = range(1, train_param.epochs + 1)
        
        ax.plot(epochs_range, history_train_loss, label='Training Loss', color='tab:blue', linewidth=2)
        ax.plot(epochs_range, history_val_loss, label='Validation Loss', color='tab:orange', linewidth=2)
        
        # Evidenzia l'epoca ottimale (minima validation loss)
        best_epoch = int(np.argmin(history_val_loss)) + 1
        min_val_loss = np.min(history_val_loss)
        ax.scatter(best_epoch, min_val_loss, color='red', s=80, zorder=5, label=f'Best Epoch: {best_epoch}')
        ax.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.5)

        ax.set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoca', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=11)
        
        fig_loss.tight_layout()
        
        # Salvataggio del grafico su disco, TensorBoard 
        loss_plot_path = os.path.join(CHECKPOINT_DIR, "loss_curves.png")
        fig_loss.savefig(loss_plot_path, dpi=300)
        writer.add_figure("Visual/Loss_Curves", fig_loss, global_step=train_param.epochs)

        plt.close(fig_loss)

        # --- SALVATAGGIO FINALE MODELLO (IMPACCHETTAMENTO MLFLOW) ---
        print("Registrazione modelli su MLflow...")
        
        mlflow.pytorch.log_model(
            pytorch_model=model, 
            name="vae_full_model",
            registered_model_name=f"VAE_{hparams.in_channels}ch"
        )
        local_vae_pack = os.path.join(OUTPUT_DIR, "VAE_full")
        mlflow.pytorch.save_model(model, path=local_vae_pack)
        
        only_decoder = OnlyDecoder(model)
        mlflow.pytorch.log_model(
            pytorch_model=only_decoder, 
            name="decoder_only_model",
            registered_model_name=f"Decoder_{hparams.in_channels}ch"
        )
        local_decoder_pack = os.path.join(OUTPUT_DIR, "Decoder_only")
        mlflow.pytorch.save_model(only_decoder, path=local_decoder_pack)

    writer.close()
    print(f"Training concluso. Checkpoint fisici in {CHECKPOINT_DIR}, log in {log_dir}")


if __name__ == "__main__":
    train()