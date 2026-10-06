from datetime import datetime
import os
from models.aekl_no_attention import AutoencoderKL
from models.ddim import DDIMSampler
from models.ddpm_v2_conditioned import DDPM
import mlflow
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils.config_aekl_v3 import get_hparams
from utils.config_train import train_config
from utils.config_unet_v3 import get_config
from utils.dataset_v3 import RadioPatchDataset
from utils.plot_new import comparison_plots_ok, denormalize_data


# --- CONFIGURAZIONE PERCORSI E DIRECTORY ---
train_cfg, _ = train_config()
hparams, _ = get_hparams()

BASE_SCRATCH = "/leonardo_scratch/large/userexternal/gvitanza/InverseSR/"
RUN_DIR = train_cfg.output_dir_ddpm
current_time = datetime.now().strftime("%b%d_%H-%M-%S")

print("Inizio configurazione MLFlow...")
db_path = "mlruns_ddpm.db"
mlflow.set_tracking_uri(f"sqlite:///{db_path}")
mlflow.set_experiment(
    f"Radio_DDPM_v2_{train_cfg.epochs}epochs_z{hparams.z_channels}_{current_time}"
)

print(f"Caricamento VAE pre-addestrato da {train_cfg.vae_path}")
vae = AutoencoderKL(embed_dim=hparams.z_channels, hparams=vars(hparams)).to(
    train_cfg.device
)
checkpoint_vae = torch.load(
    train_cfg.vae_path, map_location=train_cfg.device, weights_only=False
)
vae.load_state_dict(checkpoint_vae["model_state_dict"])
vae.eval()


def train():
    CHECKPOINT_DIR = os.path.join(
        BASE_SCRATCH,
        f"ddpm_{train_cfg.cond_key}_{hparams.z_channels}_{train_cfg.epochs}epochs_{train_cfg.norm_mode}_{current_time}",
    )
    TB_LOG_DIR = train_cfg.tensor_board_logger_ddpm
    log_dir = f"{TB_LOG_DIR}/run_{current_time}_lr_{train_cfg.learning_rate}_z{hparams.z_channels}"

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 1. Configurazione UNet
    unet_cfg, _ = get_config()

    use_mask_channel = unet_cfg["params"]["use_mask_channel"]
    in_ch = hparams.z_channels + (1 if use_mask_channel else 0)

    unet_cfg["params"]["in_channels"] = in_ch
    unet_cfg["params"]["out_channels"] = hparams.z_channels  # Output ricostruisce solo z
    unet_cfg["params"].pop("out_channels_unet", None)
    unet_cfg["params"].pop("in_channels_unet", None)

    # 2. Dataset e DataLoader (TRAIN & VALIDATION)
    train_dataset = RadioPatchDataset(
        data_dir=os.path.join(train_cfg.data_dir, "train"),
        catalogue_path=train_cfg.catalogue_path,
        in_channels=hparams.in_channels,
        norm_mode=train_cfg.norm_mode,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_dataset = RadioPatchDataset(
        data_dir=os.path.join(train_cfg.data_dir, "val"),
        in_channels=hparams.in_channels,
        norm_mode=train_cfg.norm_mode,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # 3. Modello DDPM Latente, Ottimizzatore e Scheduler
    model = DDPM(
        unet_config=unet_cfg,
        conditioning_key=train_cfg.cond_key,
        learn_logvar=True,
    ).to(train_cfg.device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=train_cfg.learning_rate
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=train_cfg.epochs,
        eta_min=1e-6
    )

    writer = SummaryWriter(log_dir=log_dir)

    # Liste per tracciare lo storico delle loss per ogni epoca
    history_train_loss = []
    history_val_loss = []

    with mlflow.start_run(run_name=f"DDPM_Training_{current_time}"):
        mlflow.log_params(vars(train_cfg))
        mlflow.log_params({f"vae_{k}": v for k, v in vars(hparams).items()})
        mlflow.log_params(
            {f"unet_{k}": v for k, v in unet_cfg["params"].items()}
        )

        for epoch in range(train_cfg.epochs):
            # ==================== FASE DI TRAINING ====================
            model.train()
            pbar = tqdm(train_dataloader, desc=f"Epoch {epoch} [Train]")
            epoch_loss = []

            for batch in pbar:
                optimizer.zero_grad()

                x_start = batch["x_0"].to(train_cfg.device)
                spatial_mask = batch["spatial_mask"].to(train_cfg.device)
                context = batch["context"].to(train_cfg.device)

                # STEP 1: Encoding Latente via VAE
                with torch.no_grad():
                    h = vae.encoder(x_start)
                    z = vae.quant_conv_mu(h)

                # STEP 2: Preparazione Condizionamento
                cond_payload = {}

                if train_cfg.cond_key in ["concat", "hybrid"]:
                    latent_shape = z.shape[2:]
                    mask_latent = F.interpolate(
                        spatial_mask,
                        size=latent_shape,
                        mode="trilinear",
                        align_corners=False,
                    )
                    cond_payload["c_concat"] = [mask_latent]

                if (
                    train_cfg.cond_key in ["crossattn", "hybrid"]
                    and context is not None
                ):
                    context_attn = (
                        context.unsqueeze(1)
                        if context.ndim == 2
                        else context
                    )
                    cond_payload["c_crossattn"] = [context_attn]

                # STEP 3: Loss e Backward
                loss, _ = model(z, cond_payload)
                loss.backward()
                optimizer.step()

                epoch_loss.append(loss.item())
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            current_lr = scheduler.get_last_lr()[0]
            scheduler.step()

            avg_train_loss = np.mean(epoch_loss)
            history_train_loss.append(avg_train_loss)  # Salviamo nello storico

            writer.add_scalar("Loss/Train_DDPM", avg_train_loss, epoch)
            writer.add_scalar("Params/LearningRate", current_lr, epoch)
            mlflow.log_metric("avg_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("learning_rate", current_lr, step=epoch)

            # ==================== FASE DI VALIDAZIONE ====================
            model.eval()
            val_losses = []

            with torch.no_grad():
                for val_batch in val_dataloader:
                    x_val = val_batch["x_0"].to(train_cfg.device)
                    mask_val = val_batch["spatial_mask"].to(train_cfg.device)
                    context_val = val_batch["context"].to(train_cfg.device)

                    h_val = vae.encoder(x_val)
                    z_val = vae.quant_conv_mu(h_val)

                    val_cond_payload = {}
                    if train_cfg.cond_key in ["concat", "hybrid"]:
                        mask_latent_val = F.interpolate(
                            mask_val,
                            size=z_val.shape[2:],
                            mode="trilinear",
                            align_corners=False,
                        )
                        val_cond_payload["c_concat"] = [mask_latent_val]

                    if (
                        train_cfg.cond_key in ["crossattn", "hybrid"]
                        and context_val is not None
                    ):
                        context_attn_val = (
                            context_val.unsqueeze(1)
                            if context_val.ndim == 2
                            else context_val
                        )
                        val_cond_payload["c_crossattn"] = [context_attn_val]

                    v_loss, _ = model(z_val, val_cond_payload)
                    val_losses.append(v_loss.item())

            avg_val_loss = np.mean(val_losses)
            history_val_loss.append(avg_val_loss)  # Salviamo nello storico

            writer.add_scalar("Loss/Val_DDPM", avg_val_loss, epoch)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)

            # ==================== GENERAZIONE E PLOT ====================
            if epoch % 20 == 0:
                with torch.no_grad():
                    val_sample_batch = next(iter(val_dataloader))
                    x_vis = val_sample_batch["x_0"][:2].to(train_cfg.device)
                    mask_vis = val_sample_batch["spatial_mask"][:2].to(train_cfg.device)
                    context_vis = val_sample_batch["context"][:2].to(train_cfg.device)

                    eval_batch_size = x_vis.shape[0]

                    h_vis = vae.encoder(x_vis)
                    z_vis = vae.quant_conv_mu(h_vis)
                    _, _, D_lat, H_lat, W_lat = z_vis.shape
                    shape = (hparams.z_channels, D_lat, H_lat, W_lat)

                    vis_cond_payload = {}
                    if train_cfg.cond_key in ["concat", "hybrid"]:
                        mask_latent_vis = F.interpolate(
                            mask_vis,
                            size=(D_lat, H_lat, W_lat),
                            mode="trilinear",
                            align_corners=False,
                        )
                        vis_cond_payload["c_concat"] = [mask_latent_vis]

                    if (
                        train_cfg.cond_key in ["crossattn", "hybrid"]
                        and context_vis is not None
                    ):
                        context_attn_vis = (
                            context_vis.unsqueeze(1)
                            if context_vis.ndim == 2
                            else context_vis
                        )
                        vis_cond_payload["c_crossattn"] = [context_attn_vis]

                    sampler = DDIMSampler(model)
                    img_noise = torch.randn(
                        (eval_batch_size, *shape), device=train_cfg.device
                    )

                    z_gen, _ = sampler.sample(
                        S=50,
                        batch_size=eval_batch_size,
                        shape=shape,
                        conditioning=vis_cond_payload,
                        first_img=img_noise,
                        eta=0.0,
                        verbose=False,
                    )

                    x_gen = vae.decode(z_gen)

                    x_real_denorm = denormalize_data(x_vis, train_cfg.norm_mode)
                    x_gen_denorm = denormalize_data(x_gen, train_cfg.norm_mode)

                    mask_sample = mask_vis[0, 0].cpu().numpy()
                    src_z, src_y, src_x = np.where(mask_sample > 0.5)
                    coords = list(zip(src_x, src_y, src_z))

                    try:
                        fig = comparison_plots_ok(
                            x_real_denorm[0],
                            x_gen_denorm[0],
                            sources_coords=coords
                        )
                        writer.add_figure(
                            "Visual/3D_Comparison", fig, global_step=epoch
                        )
                        plt.close(fig)
                    except Exception as e:
                        print(
                            f"[Warning] Errore durante la generazione del plot all'epoca {epoch}: {e}"
                        )
                        plt.close("all")
                model.train()

            # --- CHECKPOINT PERIODICO ---
            if (epoch + 1) % 20 == 0 or (epoch + 1) == train_cfg.epochs:
                ckpt_path = os.path.join(
                    CHECKPOINT_DIR, f"ddpm_ep{epoch+1}.pth"
                )
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "val_loss": avg_val_loss,
                    },
                    ckpt_path,
                )

        # --- SALVATAGGIO FINALE ---
        print("Registrazione modello DDPM finale...")
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="ddpm",
            registered_model_name=f"DDPM_{hparams.z_channels}ch",
        )

        local_model_path = os.path.join(CHECKPOINT_DIR, "ddpm_final_model") # <-- Corretto RUN_DIR in CHECKPOINT_DIR
        mlflow.pytorch.save_model(model, path=local_model_path)
        
        print("Generazione grafico Train vs Validation Loss...")
        fig_loss, ax = plt.subplots(figsize=(10, 6))
        epochs_range = range(1, train_cfg.epochs + 1)
        
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
        
        # Salvataggio del grafico su disco e su TensorBoard
        loss_plot_path = os.path.join(CHECKPOINT_DIR, "loss_curves.png")
        fig_loss.savefig(loss_plot_path, dpi=300)
        writer.add_figure("Visual/Loss_Curves", fig_loss, global_step=train_cfg.epochs)

        plt.close(fig_loss)

        writer.close()
        print(f"Training concluso. Checkpoint in {CHECKPOINT_DIR}")

if __name__ == "__main__":
    train()