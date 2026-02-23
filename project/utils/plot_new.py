from typing import List, Optional
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Configurazione default per Astro
DEFAULT_CMAP = "inferno"  # O 'viridis', 'magma', 'cividis'
BG_COLOR = "black"       # Spesso i plot astro sono più belli su sfondo scuro
def draw_corrupted_images(
    img1: np.ndarray, img2: np.ndarray, img3: np.ndarray, img4: np.ndarray, ssim_: float
) -> np.ndarray:
    si, sj, sk = img1.shape
    si_, sj_, sk_ = img3.shape
    img1_row1 = np.rot90(img1[:, :, sk // 2], -1)
    img2_row1 = np.rot90(img2[:, :, sk // 2], -1)
    img3_row1 = np.rot90(img3[:, :, sk_ // 2], -1)
    img4_row1 = np.rot90(img4[:, :, sk_ // 2], -1)
    img1_row2 = np.rot90(img1[:, sj // 2, :], -1)
    img2_row2 = np.rot90(img2[:, sj // 2, :], -1)
    img3_row2 = np.rot90(img3[:, sj_ // 2, :], -1)
    img4_row2 = np.rot90(img4[:, sj_ // 2, :], -1)
    img1_row3 = np.rot90(img1[si // 2, :, :], -1)
    img2_row3 = np.rot90(img2[si // 2, :, :], -1)
    img3_row3 = np.rot90(img3[si_ // 2, :, :], -1)
    img4_row3 = np.rot90(img4[si_ // 2, :, :], -1)
    imgs_list = [
        img1_row1,
        img2_row1,
        img3_row1,
        img4_row1,
        img1_row2,
        img2_row2,
        img3_row2,
        img4_row2,
        img1_row3,
        img2_row3,
        img3_row3,
        img4_row3,
    ]
    titles_list = [
        "Reconstructed Image",
        "Original Corrupted",
        "Reconstructed Image (downsampled)",
        "Original Corrupted (downsampled)",
    ]

    fig = plt.figure(figsize=(16, 18))
    nrows, ncols = 3, 4
    gs = gridspec.GridSpec(nrows=nrows, ncols=ncols)
    for idx in range(nrows):
        for jdx in range(ncols):
            ax = plt.subplot(gs[idx * ncols + jdx])
            ax.imshow(imgs_list[idx * ncols + jdx], cmap="gray")
            ax.grid(False)
            ax.invert_xaxis()
            ax.invert_yaxis()
            ax.set_xticks([])
            ax.set_yticks([])
            if idx == 0:
                ax.set_title(titles_list[idx * ncols + jdx])

    plt.tight_layout()
    fig.suptitle(f"SSIM: {ssim_:.4f}", x=0.48, y=0.99, fontsize=12)
    return fig


def draw_images_for_variational_inference(
    corrupted: np.ndarray, target: np.ndarray, synth_imgs: np.ndarray, ssim_: float
) -> np.ndarray:
    _, _, si, sj, sk = target.shape
    _, _, si_, sj_, sk_ = corrupted.shape
    print(f"synth_imgs.shape: {synth_imgs.shape}")
    img1_row1 = np.rot90(corrupted[0, 0, :, :, sk_ // 2], -1)
    img2_row1 = np.rot90(target[0, 0, :, :, sk // 2], -1)
    img3_row1 = np.rot90(synth_imgs[0, 0, :, :, sk // 2], -1)
    img4_row1 = np.rot90(synth_imgs[1, 0, :, :, sk // 2], -1)
    img5_row1 = np.rot90(synth_imgs[2, 0, :, :, sk // 2], -1)
    img6_row1 = np.rot90(synth_imgs[3, 0, :, :, sk // 2], -1)
    img1_row2 = np.rot90(corrupted[0, 0, :, sj_ // 2, :], -1)
    img2_row2 = np.rot90(target[0, 0, :, sj // 2, :], -1)
    img3_row2 = np.rot90(synth_imgs[0, 0, :, sj // 2, :], -1)
    img4_row2 = np.rot90(synth_imgs[1, 0, :, sj // 2, :], -1)
    img5_row2 = np.rot90(synth_imgs[2, 0, :, sj // 2, :], -1)
    img6_row2 = np.rot90(synth_imgs[3, 0, :, sj // 2, :], -1)
    img1_row3 = np.rot90(corrupted[0, 0, si_ // 2, :, :], -1)
    img2_row3 = np.rot90(target[0, 0, si // 2, :, :], -1)
    img3_row3 = np.rot90(synth_imgs[0, 0, si // 2, :, :], -1)
    img4_row3 = np.rot90(synth_imgs[1, 0, si // 2, :, :], -1)
    img5_row3 = np.rot90(synth_imgs[2, 0, si // 2, :, :], -1)
    img6_row3 = np.rot90(synth_imgs[3, 0, si // 2, :, :], -1)
    imgs_list = [
        img1_row1,
        img2_row1,
        img3_row1,
        img4_row1,
        img5_row1,
        img6_row1,
        img1_row2,
        img2_row2,
        img3_row2,
        img4_row2,
        img5_row2,
        img6_row2,
        img1_row3,
        img2_row3,
        img3_row3,
        img4_row3,
        img5_row3,
        img6_row3,
    ]
    titles_list = [
        "Corrupted Image",
        "Target Image",
        "Est. Mean",
        "Sample 1",
        "Sample 2",
        "Sample 3",
    ]

    fig = plt.figure(figsize=(24, 18))
    nrows, ncols = 3, 6
    gs = gridspec.GridSpec(nrows=nrows, ncols=ncols)
    for idx in range(nrows):
        for jdx in range(ncols):
            ax = plt.subplot(gs[idx * ncols + jdx])
            ax.imshow(imgs_list[idx * ncols + jdx], cmap="gray")
            ax.grid(False)
            ax.invert_xaxis()
            ax.invert_yaxis()
            ax.set_xticks([])
            ax.set_yticks([])
            if idx == 0:
                ax.set_title(titles_list[idx * ncols + jdx])

    plt.tight_layout()
    fig.suptitle(f"SSIM: {ssim_:.4f}", x=0.48, y=0.99, fontsize=12)
    return fig


def draw_images(
    img1: np.ndarray,
    img2: np.ndarray,
    ssim_: float,
    titles_list: List[str] = [
        "Reconstructed Image",
        "Original Corrupted",
    ],
) -> np.ndarray:
    si, sj, sk = img1.shape
    img1_row1 = np.rot90(img1[:, :, sk // 2], -1)
    img2_row1 = np.rot90(img2[:, :, sk // 2], -1)
    img1_row2 = np.rot90(img1[:, sj // 2, :], -1)
    img2_row2 = np.rot90(img2[:, sj // 2, :], -1)
    img1_row3 = np.rot90(img1[si // 2, :, :], -1)
    img2_row3 = np.rot90(img2[si // 2, :, :], -1)
    imgs_list = [
        img1_row1,
        img2_row1,
        img1_row2,
        img2_row2,
        img1_row3,
        img2_row3,
    ]

    fig = plt.figure(figsize=(8, 18))
    nrows, ncols = 3, 2
    gs = gridspec.GridSpec(nrows=nrows, ncols=ncols)
    for idx in range(nrows):
        for jdx in range(ncols):
            ax = plt.subplot(gs[idx * ncols + jdx])
            ax.imshow(imgs_list[idx * ncols + jdx], cmap="gray")
            ax.grid(False)
            ax.invert_xaxis()
            ax.invert_yaxis()
            ax.set_xticks([])
            ax.set_yticks([])
            if idx == 0:
                ax.set_title(titles_list[idx * ncols + jdx])

    plt.tight_layout()
    fig.suptitle(f"SSIM: {ssim_:.4f}", x=0.48, y=0.99, fontsize=12)
    return fig


def draw_img(img: np.ndarray, title: str, step: str, output_folder: Path) -> None:
    fig, ax = plt.subplots()
    si, sj, sk = img.shape
    img_slice = np.rot90(img[:, sj // 2, :], -1)
    ax.imshow(img_slice, cmap="gray")
    ax.axis("off")
    fig.savefig(
        output_folder / f"{step}_{title}.png",
        bbox_inches="tight",
        pad_inches=0,
        format="png",
        dpi=300,
    )
    # close
    plt.close(fig)


def normalize_for_plot(img: np.ndarray):
    """Normalizza l'immagine tra 0 e 1 per il plotting, gestendo NaN e Infinity."""
    img = np.nan_to_num(img) # Sostituisce NaN con 0
    vmin, vmax = img.min(), img.max()
    if vmax - vmin < 1e-6:
        return img
    return (img - vmin) / (vmax - vmin)

def plot_orthogonal_cuts(
    cube: np.ndarray, 
    title: str = "Astro Object", 
    save_path: Optional[Path] = None,
    ssim: Optional[float] = None
) -> plt.Figure:
    """
    Visualizza i tre tagli ortogonali di un datacube astrofisico:
    1. Piano spaziale (XY) - Sommato lungo l'asse spettrale (Moment 0)
    2. Spettrale X-Z (Posizione-Velocità lungo RA)
    3. Spettrale Y-Z (Posizione-Velocità lungo Dec)
    """
    # Assumiamo forma [Channels, Depth(Vel), Height(Dec), Width(RA)]
    # O semplicemente [Depth, Height, Width] se monocromatico.
    if len(cube.shape) == 4:
        cube = cube[0] # Rimuoviamo dimensione canale se presente
        
    nz, ny, nx = cube.shape
    
    # --- Calcolo dei tagli (Slices) ---
    
    # 1. Mappa Spaziale (Moment 0): Somma tutto il flusso lungo l'asse Z (Velocità/Freq)
    #    Questo mostra l'oggetto intero nel cielo.
    img_spatial = np.sum(cube, axis=0) 
    
    # 2. Taglio Spettrale RA (Slice centrale):
    #    Tagliamo a metà della declinazione per vedere il profilo di velocità
    img_spectral_ra = cube[:, ny // 2, :] 
    
    # 3. Taglio Spettrale Dec:
    img_spectral_dec = cube[:, :, nx // 2]

    imgs = [img_spatial, img_spectral_ra, img_spectral_dec]
    titles = ["Spatial (Moment 0)", "Spectral (PV - RA)", "Spectral (PV - Dec)"]
    
    # --- Plotting ---
    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(1, 3)
    
    for i, img in enumerate(imgs):
        ax = plt.subplot(gs[i])
        
        # origin='lower' è CRUCIALE per i FITS, altrimenti l'immagine è capovolta
        im = ax.imshow(img, cmap=DEFAULT_CMAP, origin='lower', aspect='auto')
        
        ax.set_title(titles[i])
        ax.set_xlabel("Pixels")
        ax.set_ylabel("Pixels")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    full_title = title
    if ssim is not None:
        full_title += f" | SSIM: {ssim:.4f}"
        
    fig.suptitle(full_title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        #plt.close(fig)
        
    return fig

def compare_cubes(
    original: np.ndarray, 
    reconstructed: np.ndarray, 
    title: str = "Comparison",
    save_path: Optional[Path] = None
):
    """
    Confronta visivamente il cubo originale e quello generato/ricostruito
    mostrando la mappa spaziale integrata (M0).
    """
    # Gestione dimensioni [C, D, H, W] -> [H, W] (somma su C e D)
    if len(original.shape) == 4:
        # Somma su canali e su asse spettrale (asse 1)
        # original[0] -> shape (D, H, W) -> sum(0) -> (H, W)
        img_orig = np.sum(original[0], axis=0)
        img_recon = np.sum(reconstructed[0], axis=0)
    else:
        img_orig = np.sum(original, axis=0)
        img_recon = np.sum(reconstructed, axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Originale
    im1 = axes[0].imshow(img_orig, cmap=DEFAULT_CMAP, origin='lower')
    axes[0].set_title("Ground Truth (Integrated)")
    plt.colorbar(im1, ax=axes[0])
    
    # Ricostruito
    im2 = axes[1].imshow(img_recon, cmap=DEFAULT_CMAP, origin='lower')
    axes[1].set_title("Generated / Reconstructed")
    plt.colorbar(im2, ax=axes[1])
    
    # Residui (Differenza)
    # Normalizziamo la differenza per vederla meglio
    diff = img_orig - img_recon
    v_max_diff = max(abs(diff.min()), abs(diff.max()))
    
    im3 = axes[2].imshow(diff, cmap="seismic", origin='lower', vmin=-v_max_diff, vmax=v_max_diff)
    axes[2].set_title("Residuals (Orig - Recon)")
    plt.colorbar(im3, ax=axes[2])

    fig.suptitle(title)
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        #plt.close(fig)
    return fig