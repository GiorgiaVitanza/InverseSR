import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

@torch.no_grad()
def quick_test_metrics(model, vae, dataloader, device, epoch=0):
    model.eval()
    vae.eval()
    
    # Prendi un batch dal validazione
    batch = next(iter(dataloader))
    x_start = batch["x_0"].to(device)       # Originale [B, 1, 128, 128, 128]
    context = batch["context"].to(device)   # Parametri fisici [B, 4]
    
    # 1. Definisci lo spazio latente (16 canali, risoluzione ridotta dal VAE)
    # Esempio: se il VAE comprime di un fattore 4, la risoluzione è 32
    latent_shape = (x_start.shape[0], 16, 32, 32, 32) 
    
    # 2. Generazione DDIM veloce (50 passi)
    # Partiamo da rumore puro e usiamo il context reale
    z_gen = model.sample(
        cond=context,
        batch_size=x_start.shape[0],
        shape=latent_shape[1:],
        ddim=True,
        ddim_steps=50
    )
    
    # 3. Decodifica VAE
    x_gen = vae.decode(z_gen)
    
    # 4. Calcolo metriche medie sul batch
    mses, ssims, psnrs = [], [], []
    
    for i in range(x_start.shape[0]):
        img_true = x_start[i, 0].cpu().numpy()
        img_gen = x_gen[i, 0].cpu().numpy()
        
        # Metriche sulle slice centrali (più veloci e indicative)
        mid = img_true.shape[0] // 2
        mses.append(mse(img_true[mid], img_gen[mid]))
        ssims.append(ssim(img_true[mid], img_gen[mid], data_range=img_true[mid].max() - img_true[mid].min()))
        psnrs.append(psnr(img_true[mid], img_gen[mid], data_range=1.0))

    # 5. Report e Log
    metrics = {
        "test/mse": np.mean(mses),
        "test/ssim": np.mean(ssims),
        "test/psnr": np.mean(psnrs)
    }
    
    print(f"\n--- TEST METRICS EPOCH {epoch} ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")
        mlflow.log_metric(k, v, step=epoch)

    # 6. Salvataggio visivo veloce
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(x_start[0, 0, x_start.shape[2]//2].cpu(), cmap='hot')
    plt.title("Originale")
    plt.subplot(1, 2, 2)
    plt.imshow(x_gen[0, 0, x_gen.shape[2]//2].cpu(), cmap='hot')
    plt.title("Generato (DDIM 50 step)")
    
    plt.savefig(f"quick_test_ep{epoch}.png")
    plt.close()
    
    model.train()
    return metrics

if __name__ == "__main__":
    
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
    unet_config, _ = get_config()
    
    dataset = RadioPatchDataset(data_dir=train_param.data_dir, catalogue_path=train_param.catalogue_path, in_channels=hparams.in_channels)
    dataloader = DataLoader(dataset, batch_size=train_param.batch_size, shuffle=True, num_workers=1, pin_memory=True, persistent_workers=True)

    # Inizializza VAE e DDPM (assumendo che siano già addestrati)
    vae = AutoencoderKL(embed_dim=hparams.z_channels, hparams=vars(hparams)).to(train_param.device)
    model = DDPM(
        unet_config=unet_config,
        in_channels=hparams.z_channels,  # Canali latenti del VAE
        context_channels=4,              # Parametri fisici
        device=train_param.device 
    )

    # Esegui il test rapido
    quick_test_metrics(model, vae, dataloader, train_param.device, epoch=0)