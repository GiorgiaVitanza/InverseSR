import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import mlflow 

from models.ddim import DDIMSampler
from project.BRGM_decoder import denormalize_data

@torch.no_grad()
def quick_test_metrics(model, vae, dataloader, device, epoch=0):
    model.eval()
    vae.eval()
    
    # 1. Inizializza il campionatore DDIM
    sampler = DDIMSampler(model)
    
    batch = next(iter(dataloader))
    x_start = batch["x_0"].to(device)
    context = batch["context"].to(device)
    
    # Parametri per il campionamento
    batch_size = 1 
    ddim_steps = 50  
    latent_size = 32 
    shape = (8, latent_size, latent_size, latent_size) # C, D, H, W
    
    img_noise = torch.randn((batch_size, *shape), device=device)
    
    print(f"Generazione DDIM ({ddim_steps} passi)...")
    
    # 2. Campionamento
    z_gen, _ = sampler.sample(
        S=ddim_steps,
        batch_size=batch_size,
        shape=shape,
        conditioning=context,
        first_img=img_noise,
        eta=0.0, 
        verbose=False
    )
    
    # 3. Decodifica
    x_gen = vae.decode(z_gen)
    if not isinstance(x_gen, torch.Tensor):
        x_gen = x_gen.sample()

    
    x_start_norm = denormalize_data(x_start, hparams)
    x_gen_norm = denormalize_data(x_gen, hparams)

    mses, ssims, psnrs = [], [], []
    
    # Prepariamo i dati per il plot (primo sample del batch)
    img_true = x_start_norm[0, 0].cpu().numpy()
    img_gen = x_gen_norm[0, 0].cpu().numpy()
    img_gen = np.clip(img_gen, 0, 1) # Clip necessario per PSNR/SSIM

    # --- CALCOLO SLICE E MOMENTO 0 ---
    mid = img_true.shape[0] // 2
    true_slice = img_true[mid]
    gen_slice = img_gen[mid]

    mom0_true = np.nansum(img_true, axis=0)
    mom0_gen = np.nansum(img_gen, axis=0)

    # 4. Calcolo Metriche (sulla slice centrale per coerenza con il tuo codice)
    mses.append(mse(true_slice, gen_slice))
    ssims.append(ssim(true_slice, gen_slice, data_range=1.0))
    psnrs.append(psnr(true_slice, gen_slice, data_range=1.0))

    metrics = {
        "test/mse": np.mean(mses),
        "test/ssim": np.mean(ssims),
        "test/psnr": np.mean(psnrs)
    }
    
    # 5. Report e Log
    print(f"\n--- TEST METRICS EPOCH {epoch} ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")
        try:
            mlflow.log_metric(k, v, step=epoch)
        except:
            pass

    # 6. Salvataggio visivo AVANZATO (2x2: Slice e Momento 0)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Righe 1: Slice
    vmax_s = np.percentile(true_slice, 99.9)
    im1 = axes[0, 0].imshow(true_slice, cmap='hot', vmin=0, vmax=vmax_s)
    axes[0, 0].set_title("Originale (Slice)")
    plt.colorbar(im1, ax=axes[0, 0])

    im2 = axes[0, 1].imshow(gen_slice, cmap='hot', vmin=0, vmax=vmax_s)
    axes[0, 1].set_title(f"Generato DDIM (Slice) Ep {epoch}")
    plt.colorbar(im2, ax=axes[0, 1])

    # Riga 2: Momento 0
    vmax_m = np.percentile(mom0_true, 99.9)
    im3 = axes[1, 0].imshow(mom0_true, cmap='hot', vmin=0, vmax=vmax_m)
    axes[1, 0].set_title("Originale (Momento 0)")
    plt.colorbar(im3, ax=axes[1, 0])

    im4 = axes[1, 1].imshow(mom0_gen, cmap='hot', vmin=0, vmax=vmax_m)
    axes[1, 1].set_title("Generato DDIM (Momento 0)")
    plt.colorbar(im4, ax=axes[1, 1])

    for ax in axes.ravel():
        ax.axis('off')

    plt.tight_layout()
    
    # Salvataggio locale e MLflow
    os.makedirs("./job_script", exist_ok=True)
    plot_path = f"./job_script/test_ddpm/test_ddpm_recon_ep{epoch}.png"
    plt.savefig(plot_path)
    
    try:
        mlflow.log_artifact(plot_path, artifact_path="plots_ddpm")
    except:
        pass
        
    plt.close()
    
    model.train()
    return metrics


if __name__ == "__main__":
    # Esempio di utilizzo
    from torch.utils.data import DataLoader
    from utils.dataset_v3 import RadioPatchDataset
    from ml_flow_train_vae_decoder import run_step
    from utils.config_train import train_config
    from utils.config_unet_v3 import get_config
    from utils.config_aekl_v3 import get_hparams
    from models.aekl_no_attention import AutoencoderKL
    from models.ddpm_v2_conditioned import DDPM

    # Setup dataset e dataloader
    train_param, _ = train_config()
    hparams, _ = get_hparams()
    unet_cfg, _ = get_config()
    unet_cfg["params"]["in_channels"] = unet_cfg["params"]["in_channels_unet"]
    unet_cfg["params"]["out_channels"] = unet_cfg["params"]["out_channels_unet"]
    unet_cfg["params"].pop("out_channels_unet", None)  # Rimuoviamo i parametri specifici del config per evitare confusione
    unet_cfg["params"].pop("in_channels_unet", None)
    
    dataset = RadioPatchDataset(data_dir=train_param.data_dir, catalogue_path=train_param.catalogue_path, in_channels=hparams.in_channels)
    dataloader = DataLoader(dataset, batch_size=train_param.batch_size, shuffle=True, num_workers=1, pin_memory=True, persistent_workers=True)

    # Inizializza VAE e DDPM (assumendo che siano già addestrati)
    vae = AutoencoderKL(embed_dim=hparams.z_channels, hparams=vars(hparams)).to(train_param.device)
    model = DDPM(
        unet_config=unet_cfg,
        conditioning_key=train_param.cond_key, 
        learn_logvar=True
    ).to(train_param.device)

    # Esegui il test rapido
    quick_test_metrics(model, vae, dataloader, train_param.device, epoch=10)