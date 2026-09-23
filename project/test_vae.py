import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.dataset_v3 import RadioPatchDataset
from models.aekl_no_attention import AutoencoderKL
from utils.config_train import train_config
from utils.config_aekl_v3 import get_hparams
from utils.plot_new import denormalize_data, comparison_plots_ok

# Import per metriche di qualità immagine

from skimage.metrics import peak_signal_noise_ratio as skimage_psnr
from skimage.metrics import structural_similarity as skimage_ssim

def compute_metrics(img_real, img_hat):
    # 1. Converti in numpy se sono ancora tensori PyTorch
    if hasattr(img_real, 'detach'):
        img_real = img_real.detach().cpu().numpy()
    if hasattr(img_hat, 'detach'):
        img_hat = img_hat.detach().cpu().numpy()
        
    arr_real = np.squeeze(img_real)
    arr_rec = np.squeeze(img_hat)
    
    # 2. Calcola il range dei dati
    data_range = float(arr_real.max() - arr_real.min())
    if data_range == 0:
        data_range = 1.0

    # 3. Determina la dimensione minima tra i lati dell'immagine/cubo
    min_side = min(arr_real.shape)
    
    # Se il lato più piccolo è minore di 7, adatta la dimensione della finestra (win_size)
    # win_size deve essere un numero dispari <= min_side
    if min_side < 7:
        win_size = min_side if min_side % 2 != 0 else min_side - 1
    else:
        win_size = 7

    # 4. Calcolo PSNR
    psnr_val = skimage_psnr(arr_real, arr_rec, data_range=data_range)

    # 5. Calcolo SSIM (Gestione speciale in caso di win_size troppo piccolo)
    if win_size < 3:
        # Se una dimensione è troppo piccola (es. 1 o 2 pixel), la SSIM spaziale standard non è applicabile
        ssim_val = 0.0
    else:
        # Se l'array ha 3 dimensioni (es. Cubo 3D: D, H, W)
        if arr_real.ndim == 3:
            ssim_val = skimage_ssim(
                arr_real, 
                arr_rec, 
                data_range=data_range, 
                win_size=win_size,
                channel_axis=0  # Tratta la prima dimensione come canale o esegui SSIM 3D
            )
        else:
            ssim_val = skimage_ssim(
                arr_real, 
                arr_rec, 
                data_range=data_range, 
                win_size=win_size
            )

    return psnr_val, ssim_val


def test(hparams, train_param):
    device = train_param.device
    os.makedirs(train_param.test_fig, exist_ok=True)
    print(f"Stai usando il dispositivo: {device}")

    # 2. Dataset di Test
    test_dataset = RadioPatchDataset(
        data_dir=train_param.test_dir, 
        catalogue_path=train_param.catalogue_path,
        in_channels=hparams.in_channels,
        norm_mode=train_param.norm_mode
    )
    
    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=2, 
        shuffle=False, 
        num_workers=1,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # 3. Caricamento Modello
    checkpoint_path = os.path.join(train_param.vae_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    hparams_dict = vars(hparams)
    model = AutoencoderKL(embed_dim=hparams.z_channels, hparams=hparams_dict).to(device)
    
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    print(f"Modello inizializzato e pesi caricati con successo da {checkpoint_path}.")
    
    model.eval()

    # 4. Loop di Test
    test_recon_loss = []
    test_psnr_list = []
    test_ssim_list = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Testing VAE")):
            x = batch["x_0"].to(device)
            
            # Forward pass
            h = model.encoder(x)
            z = model.quant_conv_mu(h)
            x_hat = model.decode(z)
            
            # MSE Loss sui dati normalizzati
            loss = F.mse_loss(x_hat, x)
            test_recon_loss.append(loss.item())

            # Denormalizzazione per metriche visive e fisiche
            x_denorm = denormalize_data(x, train_param.norm_mode)
            x_hat_denorm = denormalize_data(x_hat, train_param.norm_mode)

            # Calcolo PSNR e SSIM sui dati denormalizzati
            batch_psnr, batch_ssim = compute_metrics(x_denorm, x_hat_denorm)
            test_psnr_list.append(batch_psnr)
            test_ssim_list.append(batch_ssim)

            # Salvataggio plot ogni 10 batch o all'ultimo
            if i % 10 == 0 or i == len(test_loader) - 1:
                fig = comparison_plots_ok(
                    x_denorm, 
                    x_hat_denorm,
                    flag='test'
                )
                fig.savefig(f"{train_param.test_fig}/test_vae_recon_batch_{i}_{train_param.norm_mode}.png")
                plt.close(fig)  # Liberiamo la memoria della figura
            
            # Fondamentale per non saturare la RAM
            del x, x_hat, h, z, x_denorm, x_hat_denorm

    print(f"\n=================== Risultati Test VAE ===================")
    print(f" Average MSE (normalized) : {np.mean(test_recon_loss):.6f}")
    print(f" Average PSNR (denorm)    : {np.mean(test_psnr_list):.2f} dB")
    print(f" Average SSIM (denorm)    : {np.mean(test_ssim_list):.4f}")
    print(f"==========================================================\n")


if __name__ == "__main__":
    hparams, _ = get_hparams()
    train_param, _ = train_config()
    test(hparams, train_param)