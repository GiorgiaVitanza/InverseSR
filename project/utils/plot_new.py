from typing import List, Optional
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from utils.const import FITS_LIMIT, FITS_MEAN, FITS_STD

# Configurazione default per Astro
DEFAULT_CMAP = "hot"  # O 'viridis', 'magma', 'cividis'
BG_COLOR = "black"

def safe_clean(arr: np.ndarray) -> np.ndarray:
    """Rimuove NaN/Inf e garantisce un array NumPy float valido per Matplotlib."""
    if hasattr(arr, 'detach'):
        arr = arr.detach().cpu().numpy()
    elif hasattr(arr, 'cpu'):
        arr = arr.cpu().numpy()
    
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    return arr

def get_safe_bounds(arr: np.ndarray, default_range: float = 1e-5) -> tuple[float, float]:
    """Calcola vmin e vmax in modo sicuro evitando vmin >= vmax."""
    vmin = float(np.min(arr))
    vmax = float(np.max(arr))
    if vmin >= vmax:
        vmax = vmin + default_range
    return vmin, vmax

def denormalize_data(x, norm_mode):
    """Denormalizza in base alla modalità scelta per tornare ai Jy/beam"""
    mode = norm_mode

    if mode == 'global_sym':
        print("Denormalizzazione globale simmetrica")
        x_phys = (x * 2.0 - 1.0) * FITS_LIMIT
        return x_phys
        
    elif mode == 'local':
        print("Denormalizzazione locale: usando valori globali come approssimazione")
        v_min, v_max = -1.47e-03, 1.52e-03
        return x * (v_max - v_min) + v_min
        
    elif mode == 'zscore':
        print("denormalizzazione z-score")
        return x * FITS_STD + FITS_MEAN
        
    return x


def comparison_plots_ok(x_real, x_gen, title_real="Originale", title_gen="Ricostruito", sources_coords=None, flag='test'):
    x_real = safe_clean(x_real)
    x_gen = safe_clean(x_gen)

    # Gestione dimensioni (Batch e Canali)
    if x_real.ndim == 5: x_real = x_real[0]
    if x_gen.ndim == 5:  x_gen  = x_gen[0]
    if x_real.ndim == 4: x_real = x_real[0] if x_real.shape[0] in [1, 3] else x_real.squeeze()
    if x_gen.ndim == 4:  x_gen  = x_gen[0] if x_gen.shape[0] in [1, 3] else x_gen.squeeze()

    # Proiezioni MIP
    mip_real_xy = np.max(x_real, axis=0) # Z
    mip_gen_xy  = np.max(x_gen, axis=0)

    mip_real_xz = np.max(x_real, axis=1) # Y
    mip_gen_xz  = np.max(x_gen, axis=1)

    mip_real_yz = np.max(x_real, axis=2) # X
    mip_gen_yz  = np.max(x_gen, axis=2)

    vmin, vmax = get_safe_bounds(mip_real_xy)

    # Creazione figura con layout ottimizzato
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), gridspec_kw={'height_ratios': [1, 1]})

    fig.suptitle(f"Reale: {title_real}  |  Generato da: {title_gen}", fontsize=13, fontweight='bold', y=0.98)

    # --- RIGA 1: REALE ---
    im = axes[0, 0].imshow(mip_real_xy, origin="lower", cmap="inferno", vmin=vmin, vmax=vmax)
    axes[0, 0].set_title("MIP XY (Vista dall'alto)")
    axes[0, 0].set_ylabel("Originale (Y)")

    axes[0, 1].imshow(mip_real_xz, origin="lower", cmap="inferno", vmin=vmin, vmax=vmax, aspect='auto')
    axes[0, 1].set_title("MIP XZ (Vista frontale)")
    axes[0, 1].set_ylabel("Z")

    axes[0, 2].imshow(mip_real_yz, origin="lower", cmap="inferno", vmin=vmin, vmax=vmax, aspect='auto')
    axes[0, 2].set_title("MIP YZ (Vista laterale)")
    axes[0, 2].set_ylabel("Z")

    # --- RIGA 2: GENERATO ---
    axes[1, 0].imshow(mip_gen_xy, origin="lower", cmap="inferno", vmin=vmin, vmax=vmax)
    axes[1, 0].set_title("MIP XY (Vista dall'alto)")
    axes[1, 0].set_ylabel("Generato (Y)")
    axes[1, 0].set_xlabel("X")

    axes[1, 1].imshow(mip_gen_xz, origin="lower", cmap="inferno", vmin=vmin, vmax=vmax, aspect='auto')
    axes[1, 1].set_title("MIP XZ (Vista frontale)")
    axes[1, 1].set_xlabel("X")

    axes[1, 2].imshow(mip_gen_yz, origin="lower", cmap="inferno", vmin=vmin, vmax=vmax, aspect='auto')
    axes[1, 2].set_title("MIP YZ (Vista laterale)")
    axes[1, 2].set_xlabel("Y")

    # Aggiungi sorgenti GT sul piano XY
    if sources_coords is not None and len(sources_coords) > 0:
        xs = [pt[0] for pt in sources_coords]
        ys = [pt[1] for pt in sources_coords]
        for ax in [axes[0, 0], axes[1, 0]]:
            ax.scatter(xs, ys, color='cyan', marker='o', s=50, facecolors='none', linewidths=1.2)

    # Colorbar posizionata sul lato destro senza coprire i plot
    fig.tight_layout(rect=[0, 0, 0.90, 0.95])
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    return fig


def draw_img_in_three_dim(img, title: str, output_folder: Path) -> None:
    img = safe_clean(img)
    img = np.squeeze(img)
    
    if img.ndim == 4:
        img = img[0]
    
    if img.ndim != 3:
        print(f"Errore: Il volume ha shape {img.shape}, ma deve essere 3D.")
        return

    si, sj, sk = img.shape
    dim_names = ["RA-Dec", "Freq-Dec", "Freq-RA"]
    
    fig, ax = plt.subplots()
    
    # PIANO 1
    img_slice1 = img[si//2, :, :]
    vmin, vmax = get_safe_bounds(img_slice1)
    ax.imshow(img_slice1, cmap="hot", origin='lower', vmin=vmin, vmax=vmax) 
    ax.axis("off")
    ax.set_title(f"{dim_names[0]} (slice {sk // 2})")
    fig.savefig(output_folder / f"{title}_{dim_names[0]}.png", bbox_inches="tight", pad_inches=0.1, dpi=300)

    # PIANO 2
    img_slice2 = img[:, sj // 2, :]
    vmin, vmax = get_safe_bounds(img_slice2)
    ax.imshow(img_slice2, cmap="hot", origin='lower', vmin=vmin, vmax=vmax)
    ax.axis("off")
    ax.set_title(f"{dim_names[1]} (slice {sj // 2})")
    fig.savefig(output_folder / f"{title}_{dim_names[1]}.png", bbox_inches="tight", pad_inches=0.1, dpi=300)

    # PIANO 3
    img_slice3 = img[:, :, sk//2]
    vmin, vmax = get_safe_bounds(img_slice3)
    ax.imshow(img_slice3, cmap="hot", origin='lower', vmin=vmin, vmax=vmax)
    ax.axis("off")
    ax.set_title(f"{dim_names[2]} (slice {si // 2})")
    fig.savefig(output_folder / f"{title}_{dim_names[2]}.png", bbox_inches="tight", pad_inches=0.1, dpi=300)

    plt.close(fig)


def draw_corrupted_images(
    img1: np.ndarray, img2: np.ndarray, img3: np.ndarray, img4: np.ndarray, ssim_: float
) -> plt.Figure:
    img1, img2, img3, img4 = safe_clean(img1), safe_clean(img2), safe_clean(img3), safe_clean(img4)

    si, sj, sk = img1.shape
    si_, sj_, sk_ = img3.shape

    imgs_list = [
        img1[:, :, sk // 2],  img2[:, :, sk // 2],  img3[:, :, sk_ // 2],  img4[:, :, sk_ // 2],
        img1[:, sj // 2, :],  img2[:, sj // 2, :],  img3[:, sj_ // 2, :],  img4[:, sj_ // 2, :],
        img1[si // 2, :, :],  img2[si // 2, :, :],  img3[si_ // 2, :, :],  img4[si_ // 2, :, :]
    ]
    titles_list = [
        "Reconstructed Image", "Original Corrupted",
        "Reconstructed Image (downsampled)", "Original Corrupted (downsampled)",
    ]

    fig = plt.figure(figsize=(16, 18))
    nrows, ncols = 3, 4
    gs = gridspec.GridSpec(nrows=nrows, ncols=ncols)
    for idx in range(nrows):
        for jdx in range(ncols):
            curr_img = imgs_list[idx * ncols + jdx]
            vmin, vmax = get_safe_bounds(curr_img)
            
            ax = plt.subplot(gs[idx * ncols + jdx])
            ax.imshow(curr_img, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            if idx == 0:
                ax.set_title(titles_list[jdx])

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
) -> plt.Figure:
    img1, img2 = safe_clean(img1), safe_clean(img2)
    si, sj, sk = img1.shape

    imgs_list = [
        img1[:, :, sk // 2], img2[:, :, sk // 2],
        img1[:, sj // 2, :], img2[:, sj // 2, :],
        img1[si // 2, :, :], img2[si // 2, :, :]
    ]

    fig = plt.figure(figsize=(8, 18))
    nrows, ncols = 3, 2
    gs = gridspec.GridSpec(nrows=nrows, ncols=ncols)
    for idx in range(nrows):
        for jdx in range(ncols):
            curr_img = imgs_list[idx * ncols + jdx]
            vmin, vmax = get_safe_bounds(curr_img)

            ax = plt.subplot(gs[idx * ncols + jdx])
            ax.imshow(curr_img, cmap=DEFAULT_CMAP, origin="lower", vmin=vmin, vmax=vmax)
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            if idx == 0:
                ax.set_title(titles_list[jdx])

    plt.tight_layout()
    fig.suptitle(f"SSIM: {ssim_:.4f}", x=0.48, y=0.99, fontsize=12)
    return fig


def draw_img(img: np.ndarray, title: str, step: str, output_folder: Path) -> None:
    img = safe_clean(img)
    fig, ax = plt.subplots()
    si, sj, sk = img.shape
    img_slice = img[si // 2, :, :]
    
    vmin, vmax = get_safe_bounds(img_slice)
    
    im_plot = ax.imshow(img_slice, cmap=DEFAULT_CMAP, origin="lower", vmin=vmin, vmax=vmax)
    
    ax.set_title(title)
    ax.set_xlabel("Pixels")
    ax.set_ylabel("Pixels")
    plt.colorbar(im_plot, ax=ax, fraction=0.046, pad=0.04)
    
    fig.savefig(
        output_folder / f"{step}_{title}.png",
        bbox_inches="tight",
        pad_inches=0,
        format="png",
        dpi=300,
    )
    plt.close(fig)


def plot_orthogonal_cuts(
    cube: np.ndarray, 
    title: str = "Astro Object", 
    save_path: Optional[Path] = None,
    ssim: Optional[float] = None
) -> plt.Figure:
    cube = safe_clean(cube)
    if len(cube.shape) == 4:
        cube = cube[0]
        
    nz, ny, nx = cube.shape
    
    img_spatial = np.sum(cube, axis=0) 
    img_spectral_ra = cube[:, ny // 2, :] 
    img_spectral_dec = cube[:, :, nx // 2]
    img_ra_dec = cube[nz // 2, :, :]

    imgs = [img_spatial, img_spectral_ra, img_spectral_dec, img_ra_dec]
    titles = ["Spatial (Moment 0)", "Spectral (Z - RA)", "Spectral (Z - Dec)", "RA-Dec (Z - Center)"]
    
    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(2, 2)
    
    for i, img in enumerate(imgs):
        ax = plt.subplot(gs[i])
        vmin, vmax = get_safe_bounds(img)
        
        im = ax.imshow(img, cmap=DEFAULT_CMAP, origin='lower', vmin=vmin, vmax=vmax)
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
        
    return fig


def compare_cubes(
    original: np.ndarray, 
    reconstructed: np.ndarray, 
    title: str = "Comparison",
    save_path: Optional[Path] = None
) -> plt.Figure:
    original = safe_clean(original)
    reconstructed = safe_clean(reconstructed)

    if original.ndim == 4: original = original[0]
    if reconstructed.ndim == 4: reconstructed = reconstructed[0]

    img_orig = np.sum(original, axis=0)
    img_recon = np.sum(reconstructed, axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Originale
    vmin1, vmax1 = get_safe_bounds(img_orig)
    im1 = axes[0].imshow(img_orig, cmap=DEFAULT_CMAP, origin='lower', vmin=vmin1, vmax=vmax1)
    axes[0].set_title("Ground Truth (Integrated)")
    plt.colorbar(im1, ax=axes[0])
    
    # Ricostruito
    vmin2, vmax2 = get_safe_bounds(img_recon)
    im2 = axes[1].imshow(img_recon, cmap=DEFAULT_CMAP, origin='lower', vmin=vmin2, vmax=vmax2)
    axes[1].set_title("Generated / Reconstructed")
    plt.colorbar(im2, ax=axes[1])
    
    # Residui
    diff = img_orig - img_recon
    v_max_diff = max(abs(np.min(diff)), abs(np.max(diff)))
    if v_max_diff == 0:
        v_max_diff = 1e-5
        
    im3 = axes[2].imshow(diff, cmap="seismic", origin='lower', vmin=-v_max_diff, vmax=v_max_diff)
    axes[2].set_title("Residuals (Orig - Recon)")
    plt.colorbar(im3, ax=axes[2])

    fig.suptitle(title)
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        
    return fig