from typing import List, Optional
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from utils.const import FITS_LIMIT, FITS_MEAN, FITS_STD

DEFAULT_CMAP = "hot"
BG_COLOR = "black"

def safe_clean(arr: np.ndarray) -> np.ndarray:
    """Rimuove NaN/Inf e garantisce un array NumPy float valido per Matplotlib."""
    if hasattr(arr, 'detach'):
        arr = arr.detach().cpu().numpy()
    elif hasattr(arr, 'cpu'):
        arr = arr.cpu().numpy()
    
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    return arr

def get_safe_bounds(
    *arrays: np.ndarray, 
    lower_percentile: float = 1.0, 
    upper_percentile: float = 99.5, 
    default_range: float = 1e-5
) -> tuple[float, float]:
    """Calcola vmin e vmax in modo sicuro usando i percentili su uno o più array."""
    valid_arrays = [a for a in arrays if a.size > 0]
    if not valid_arrays:
        return 0.0, default_range

    all_values = np.concatenate([a.ravel() for a in valid_arrays])
    
    vmin = float(np.percentile(all_values, lower_percentile))
    vmax = float(np.percentile(all_values, upper_percentile))
    
    if vmin >= vmax:
        vmax = vmin + default_range
        
    return vmin, vmax

def denormalize_data(x, norm_mode: str, patch_stats: dict = None):
    """
    Denormalizza x riportandolo alla scala fisica (Jy/beam).
    Supporta tutte le modalità: global_sym, global_robust, global_arcsinh, local, zscore.
    """
    if patch_stats is None:
        patch_stats = {}

    if norm_mode == 'global_sym':
        limit = patch_stats.get('limit', FITS_LIMIT)
        return (x * 2.0 - 1.0) * limit

    elif norm_mode == 'global_robust':
        p_min = patch_stats.get('p_min', -1.0559e-05)
        p_max = patch_stats.get('p_max', 1.8725e-05)
        return x * (p_max - p_min + 1e-8) + p_min

    elif norm_mode == 'global_arcsinh':
        p_min = patch_stats.get('p_min', -1.0559e-05)
        p_max = patch_stats.get('p_max', 1.8725e-05)
        
        p_min_arcsinh = patch_stats.get('p_min_arcsinh', float(np.arcsinh(p_min)))
        p_max_arcsinh = patch_stats.get('p_max_arcsinh', float(np.arcsinh(p_max)))

        # 1. Inversa dello Scaling Min-Max Arcsinh -> [p_min_arcsinh, p_max_arcsinh]
        x_arcsinh = x * (p_max_arcsinh - p_min_arcsinh + 1e-8) + p_min_arcsinh
        
        # 2. Inversa dell'Arcsinh (np.sinh / torch.sinh)
        if hasattr(x_arcsinh, 'sinh'):
            return np.sinh(x_arcsinh.cpu().numpy()) if hasattr(x_arcsinh, 'cpu') else np.sinh(x_arcsinh)
        return np.sinh(x_arcsinh)

    elif norm_mode == 'local':
        p_min = patch_stats.get('p_min', -1.47e-03)
        p_max = patch_stats.get('p_max', 1.52e-03)
        return x * (p_max - p_min + 1e-8) + p_min

    elif norm_mode == 'zscore':
        mean = patch_stats.get('mean', FITS_MEAN)
        std = patch_stats.get('std', FITS_STD)
        return x * std + mean

    return x

    
def comparison_plots_ok(
    x_real, 
    x_gen, 
    x_lr=None, 
    title_real="Originale", 
    title_gen="Ricostruito", 
    title_lr="Bassa Risoluzione", 
    sources_coords=None
):
    # 1. Sanitizzazione input
    x_real = safe_clean(x_real)
    x_gen = safe_clean(x_gen)
    if x_lr is not None:
        x_lr = safe_clean(x_lr)

    # Riduzione dimensioni per x_real e x_gen
    if x_real.ndim == 5: x_real = x_real[0]
    if x_gen.ndim == 5:  x_gen  = x_gen[0]
    if x_real.ndim == 4: x_real = x_real[0] if x_real.shape[0] in [1, 3] else x_real.squeeze()
    if x_gen.ndim == 4:  x_gen  = x_gen[0] if x_gen.shape[0] in [1, 3] else x_gen.squeeze()

    # Proiezioni MIP per Reale e Generato
    mip_real_xy, mip_real_xz, mip_real_yz = np.max(x_real, axis=0), np.max(x_real, axis=1), np.max(x_real, axis=2)
    mip_gen_xy,  mip_gen_xz,  mip_gen_yz  = np.max(x_gen, axis=0),  np.max(x_gen, axis=1),  np.max(x_gen, axis=2)

    # Concatenazione dati per i limiti visivi
    all_volumes = [mip_real_xy.ravel(), mip_real_xz.ravel(), mip_real_yz.ravel()]

    # Gestione Proiezioni MIP per Low Resolution (se presente)
    has_lr = x_lr is not None
    if has_lr:
        if x_lr.ndim == 5: x_lr = x_lr[0]
        if x_lr.ndim == 4: x_lr = x_lr[0] if x_lr.shape[0] in [1, 3] else x_lr.squeeze()
        
        mip_lr_xy = np.max(x_lr, axis=0)
        mip_lr_xz = np.max(x_lr, axis=1)
        mip_lr_yz = np.max(x_lr, axis=2)
        all_volumes.extend([mip_lr_xy.ravel(), mip_lr_xz.ravel(), mip_lr_yz.ravel()])

    all_real = np.concatenate(all_volumes)
    vmin, vmax = get_safe_bounds(all_real, lower_percentile=1.0, upper_percentile=99.8)

    # Setup della griglia (3 righe se x_lr è fornito, altrimenti 2 righe)
    n_rows = 3 if has_lr else 2
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4 * n_rows), gridspec_kw={'height_ratios': [1] * n_rows})
    
    main_title = f"Reale: {title_real}  |  Generato: {title_gen}"
    if has_lr:
        main_title = f"Reale: {title_real}  |  LR: {title_lr}  |  Generato: {title_gen}"
    fig.suptitle(main_title, fontsize=13, fontweight='bold', y=0.99)

    # RIGA 1: REALE
    im = axes[0, 0].imshow(mip_real_xy, origin="lower", cmap="inferno", vmin=vmin, vmax=vmax)
    axes[0, 0].set_title("MIP XY (Vista dall'alto)")
    axes[0, 0].set_ylabel("Originale (Y)")

    axes[0, 1].imshow(mip_real_xz, origin="lower", cmap="inferno", vmin=vmin, vmax=vmax, aspect='auto')
    axes[0, 1].set_title("MIP XZ (Vista frontale)")
    axes[0, 1].set_ylabel("Z")

    axes[0, 2].imshow(mip_real_yz, origin="lower", cmap="inferno", vmin=vmin, vmax=vmax, aspect='auto')
    axes[0, 2].set_title("MIP YZ (Vista laterale)")
    axes[0, 2].set_ylabel("Z")

    # RIGA MID: LOW RESOLUTION (Se presente)
    gen_row_idx = 1
    if has_lr:
        gen_row_idx = 2
        axes[1, 0].imshow(mip_lr_xy, origin="lower", cmap="inferno", vmin=vmin, vmax=vmax)
        axes[1, 0].set_title("MIP XY (Vista dall'alto)")
        axes[1, 0].set_ylabel(f"{title_lr} (Y)")

        axes[1, 1].imshow(mip_lr_xz, origin="lower", cmap="inferno", vmin=vmin, vmax=vmax, aspect='auto')
        axes[1, 1].set_title("MIP XZ (Vista frontale)")
        axes[1, 1].set_ylabel("Z")

        axes[1, 2].imshow(mip_lr_yz, origin="lower", cmap="inferno", vmin=vmin, vmax=vmax, aspect='auto')
        axes[1, 2].set_title("MIP YZ (Vista laterale)")
        axes[1, 2].set_ylabel("Z")

    # RIGA FINALE: GENERATO
    axes[gen_row_idx, 0].imshow(mip_gen_xy, origin="lower", cmap="inferno", vmin=vmin, vmax=vmax)
    axes[gen_row_idx, 0].set_title("MIP XY (Vista dall'alto)")
    axes[gen_row_idx, 0].set_ylabel("Generato (Y)")
    axes[gen_row_idx, 0].set_xlabel("X")

    axes[gen_row_idx, 1].imshow(mip_gen_xz, origin="lower", cmap="inferno", vmin=vmin, vmax=vmax, aspect='auto')
    axes[gen_row_idx, 1].set_title("MIP XZ (Vista frontale)")
    axes[gen_row_idx, 1].set_xlabel("X")

    axes[gen_row_idx, 2].imshow(mip_gen_yz, origin="lower", cmap="inferno", vmin=vmin, vmax=vmax, aspect='auto')
    axes[gen_row_idx, 2].set_title("MIP YZ (Vista laterale)")
    axes[gen_row_idx, 2].set_xlabel("Y")

    # Overlay coordinate sorgenti (se fornite)
    if sources_coords is not None and len(sources_coords) > 0:
        xs = [pt[0] for pt in sources_coords]
        ys = [pt[1] for pt in sources_coords]
        plot_axes = [axes[i, 0] for i in range(n_rows)]
        for ax in plot_axes:
            ax.scatter(xs, ys, color='cyan', marker='o', s=50, facecolors='none', linewidths=1.2)

    fig.tight_layout(rect=[0, 0, 0.90, 0.96])
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
    
    vmin, vmax = get_safe_bounds(img)
    
    fig, ax = plt.subplots()
    
    img_slice1 = img[si//2, :, :]
    ax.imshow(img_slice1, cmap="hot", origin='lower', vmin=vmin, vmax=vmax) 
    ax.axis("off")
    ax.set_title(f"{dim_names[0]} (slice {si // 2})")
    fig.savefig(output_folder / f"{title}_{dim_names[0]}.png", bbox_inches="tight", pad_inches=0.1, dpi=300)

    img_slice2 = img[:, sj // 2, :]
    ax.imshow(img_slice2, cmap="hot", origin='lower', vmin=vmin, vmax=vmax)
    ax.axis("off")
    ax.set_title(f"{dim_names[1]} (slice {sj // 2})")
    fig.savefig(output_folder / f"{title}_{dim_names[1]}.png", bbox_inches="tight", pad_inches=0.1, dpi=300)

    img_slice3 = img[:, :, sk//2]
    ax.imshow(img_slice3, cmap="hot", origin='lower', vmin=vmin, vmax=vmax)
    ax.axis("off")
    ax.set_title(f"{dim_names[2]} (slice {sk // 2})")
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

    vmin, vmax = get_safe_bounds(img2, img4)

    fig = plt.figure(figsize=(16, 18))
    nrows, ncols = 3, 4
    gs = gridspec.GridSpec(nrows=nrows, ncols=ncols)
    for idx in range(nrows):
        for jdx in range(ncols):
            curr_img = imgs_list[idx * ncols + jdx]
            
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

    vmin, vmax = get_safe_bounds(img2)

    fig = plt.figure(figsize=(8, 18))
    nrows, ncols = 3, 2
    gs = gridspec.GridSpec(nrows=nrows, ncols=ncols)
    for idx in range(nrows):
        for jdx in range(ncols):
            curr_img = imgs_list[idx * ncols + jdx]

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
    
    vmin, vmax = get_safe_bounds(img)
    
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

    if original.ndim == 5: original = original[0]
    if reconstructed.ndim == 5: reconstructed = reconstructed[0]
    if original.ndim == 4: original = original[0]
    if reconstructed.ndim == 4: reconstructed = reconstructed[0]

    img_orig = np.sum(original, axis=0)
    img_recon = np.sum(reconstructed, axis=0)

    vmin, vmax = get_safe_bounds(img_orig, lower_percentile=1.0, upper_percentile=99.8)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    im1 = axes[0].imshow(img_orig, cmap=DEFAULT_CMAP, origin='lower', vmin=vmin, vmax=vmax)
    axes[0].set_title("Ground Truth (Integrated - Mom 0)")
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    im2 = axes[1].imshow(img_recon, cmap=DEFAULT_CMAP, origin='lower', vmin=vmin, vmax=vmax)
    axes[1].set_title("Generated / Reconstructed")
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    diff = img_orig - img_recon
    _, v_max_diff = get_safe_bounds(np.abs(diff), lower_percentile=0.0, upper_percentile=99.5)
        
    im3 = axes[2].imshow(diff, cmap="seismic", origin='lower', vmin=-v_max_diff, vmax=v_max_diff)
    axes[2].set_title("Residuals (Orig - Recon)")
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
    return fig