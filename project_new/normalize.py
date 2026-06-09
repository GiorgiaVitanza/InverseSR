import matplotlib.pyplot as plt
import torch
import numpy as np
import os
from torch.utils.data import DataLoader

# Statistiche dal tuo cubo FITS
FITS_MAX = 1.52088422e-03
FITS_MIN = -1.47367257e-03
FITS_STD = 3.11374637e-05
LIMIT = 1.6e-03 # Margine di sicurezza

def apply_norm(data, mode='global_sym'):
    if mode == 'local':
        # Quella che usi ora: ogni patch ha la sua scala
        p_min = data.min()
        p_max = np.percentile(data, 99.8)
        return (data - p_min) / (p_max - p_min + 1e-8)
    
    elif mode == 'global_sym':
        # Consigliata: Zero fisico = 0.5, range [0, 1]
        x_scaled = data / LIMIT
        return (x_scaled + 1.0) / 2.0
    
    elif mode == 'zscore':
        # Media 0, Std 1 (poi compressa per visualizzazione)
        return (data / FITS_STD) 

def visualize_normalization_comparison(data_path, num_samples=2):
    # Carichiamo un file raw .npy per testare le normalizzazioni
    # Nota: carichiamo il dato "fisico" non ancora normalizzato
    raw_data = np.load(data_path).astype(np.float32)
    if raw_data.ndim > 3: raw_data = raw_data.squeeze()
    
    modes = ['local', 'global_sym', 'zscore']
    fig, axes = plt.subplots(len(modes), 2, figsize=(15, 5 * len(modes)))

    for row, mode in enumerate(modes):
        # Applichiamo la normalizzazione
        norm_cube = apply_norm(raw_data, mode=mode)
        
        # Slice centrale e Momento 0
        slice_idx = norm_cube.shape[0] // 2
        central_slice = norm_cube[slice_idx, :, :]
        mom0 = np.sum(norm_cube, axis=0)
        
        # Plot Slice
        im1 = axes[row, 0].imshow(central_slice, cmap='RdBu_r' if mode != 'local' else 'hot', origin='lower')
        axes[row, 0].set_title(f"MODALITÀ: {mode.upper()} - Slice Centrale")
        fig.colorbar(im1, ax=axes[row, 0])
        
        # Plot Momento 0
        im2 = axes[row, 1].imshow(mom0, cmap='hot', origin='lower')
        axes[row, 1].set_title(f"Momento 0 (Somma Z)")
        fig.colorbar(im2, ax=axes[row, 1])

    plt.tight_layout()
    plt.savefig('comparison_norms.png')
    print("Confronto salvato in: comparison_norms.png")

# --- TEST DIRETTO SU UNA PATCH ---
# Sostituisci con un percorso reale di una tua patch .npy
patch_test = "/leonardo_scratch/large/userexternal/gvitanza/InverseSR/data/inputs/128x128x128_stride128/npy_patches/patch_000000.npy"
if os.path.exists(patch_test):
    visualize_normalization_comparison(patch_test)