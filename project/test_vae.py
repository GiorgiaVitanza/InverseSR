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
from BRGM_decoder import denormalize_data


def test():
    # 1. Caricamento Configurazioni
    train_param, _ = train_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Stai usando il dispositivo: {device}")
    hparams, _ = get_hparams()

    # 2. Dataset di Test
    test_dataset = RadioPatchDataset(
        data_dir=train_param.test_dir, 
        catalogue_path=train_param.catalogue_path,
        in_channels=hparams.in_channels 
    )
    
    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=2, 
        shuffle=False, 
        num_workers=1, # Aumentato per Leonardo
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
    print("Modello inizializzato e pesi caricati con successo.")
    
    model.eval()

    # 4. Loop di Test
    test_recon_loss = []
    
    # Assicurati che la cartella esista
    os.makedirs("./job_script", exist_ok=True)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader)):
            x = batch["x_0"].to(device)
            
            # Forward pass
            h = model.encoder(x)
            z = model.quant_conv_mu(h)
            x_hat = model.decode(z)
            
            loss = F.mse_loss(x_hat, x)
            test_recon_loss.append(loss.item())

            # Salvataggio plot ogni 20 batch o all'ultimo
            if i % 20 == 0 or i == len(test_loader) - 1:
                # Applichiamo denormalizzazione per il plot
                x_plot = denormalize_data(x, hparams)
                x_hat_plot = denormalize_data(x_hat, hparams)
                
                save_comparison_advanced(
                    x_plot, 
                    x_hat_plot, 
                    f"./job_script/test_vae/test_vae_recon_batch_{i}.png"
                )
            
            # Fondamentale per non saturare la RAM
            del x, x_hat, h, z

    print(f"--- Risultati Test ---")
    print(f"Average MSE: {np.mean(test_recon_loss):.6f}")

def save_comparison_advanced(orig, recon, filename):
    """
    Crea un plot 2x2:
    - Slice centrale (Orig vs Recon)
    - Momento 0 (Orig vs Recon)
    """
    # Prendiamo solo il primo sample del batch per il plot
    # orig/recon shape: [B, C, Z, Y, X] -> prendiamo [0, 0]
    img_orig = orig[0, 0].cpu().numpy()
    img_recon = recon[0, 0].cpu().numpy()

    # 1. Slice Centrale
    mid_z = img_orig.shape[0] // 2
    slice_orig = img_orig[mid_z]
    slice_recon = img_recon[mid_z]

    # 2. Momento 0 (Integrazione lungo l'asse della profondità)
    # Usiamo nansum per sicurezza se ci fossero pixel mancanti
    mom0_orig = np.nansum(img_orig, axis=0)
    mom0_recon = np.nansum(img_recon, axis=0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # --- RIGA 1: SLICE ---
    # Calcoliamo vmax sul 99.9 percentile dell'originale per contrasto ottimale
    vmax_s = np.percentile(slice_orig, 99.9)
    
    im1 = axes[0, 0].imshow(slice_orig, cmap='hot', vmin=0, vmax=vmax_s)
    axes[0, 0].set_title(f"Originale (Slice Z={mid_z})")
    plt.colorbar(im1, ax=axes[0, 0])

    im2 = axes[0, 1].imshow(slice_recon, cmap='hot', vmin=0, vmax=vmax_s)
    axes[0, 1].set_title("Ricostruito (Slice)")
    plt.colorbar(im2, ax=axes[0, 1])

    # --- RIGA 2: MOMENTO 0 ---
    vmax_m = np.percentile(mom0_orig, 99.9)
    
    im3 = axes[1, 0].imshow(mom0_orig, cmap='hot', vmin=0, vmax=vmax_m)
    axes[1, 0].set_title("Originale (Momento 0)")
    plt.colorbar(im3, ax=axes[1, 0])

    im4 = axes[1, 1].imshow(mom0_recon, cmap='hot', vmin=0, vmax=vmax_m)
    axes[1, 1].set_title("Ricostruito (Momento 0)")
    plt.colorbar(im4, ax=axes[1, 1])

    for ax in axes.ravel():
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    print(f"Plot avanzato salvato in: {filename}")
    plt.close(fig)

if __name__ == "__main__":
    test()