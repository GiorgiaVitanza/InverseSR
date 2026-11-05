from typing import List, Optional
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import ScalarFormatter

from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
DEFAULT_CMAP = "hot"
BG_COLOR = "black"


def denormalize_data(x, norm_mode: str, patch_stats: dict = None):
    """
    Denormalizes x back to physical scale (Jy/beam).
    Supports all modes: global_robust, global_arcsinh, local, zscore.
    """
    if patch_stats is None:
        patch_stats = {}
    def prep_stat(key):
        if key not in patch_stats or patch_stats[key] is None:
            return None
        val = patch_stats[key]
        
        if isinstance(val, torch.Tensor):
            val = val.to(device=x.device, dtype=x.dtype)
            
            # Se la dimensione del batch di val è maggiore di x, taglia il batch
            if val.ndim > 0 and val.shape[0] > x.shape[0]:
                val = val[:x.shape[0]]
                
            # Aggiunge dimensioni singleton in coda ([2] -> [2, 1, 1, 1, 1])
            # affinché il rank di val sia identico a quello di x per un broadcasting corretto
            while val.ndim < x.ndim:
                val = val.unsqueeze(-1)
                
        return val
    if norm_mode == 'global_robust':
        p_min = prep_stat('p5')
        p_max = prep_stat('p95')
        return x * (p_max - p_min + 1e-8) + p_min

    elif norm_mode == 'global':
        p_min = prep_stat('min')
        p_max = prep_stat('max')
        return x * (p_max - p_min + 1e-8) + p_min

    elif norm_mode == 'global_arcsinh':
        p_min_arcsinh = prep_stat('p5_arcsinh')
        p_max_arcsinh = prep_stat('p95_arcsinh')

        # 1. Inverse Min-Max Arcsinh Scaling -> [p_min_arcsinh, p_max_arcsinh]
        x_arcsinh = x * (p_max_arcsinh - p_min_arcsinh + 1e-8) + p_min_arcsinh
        
        # 2. Inverse Arcsinh (sinh)
        if isinstance(x_arcsinh, torch.Tensor):
            return torch.sinh(x_arcsinh)
        return np.sinh(x_arcsinh)
    
    elif norm_mode == 'local':
        p_min = float(x.min())
        if isinstance(x, np.ndarray):
            p_max = float(np.quantile(x, 0.99))
        else:
            p_max = float(torch.quantile(x, 0.99))
        
        return x * (p_max - p_min + 1e-8) + p_min

    elif norm_mode == 'zscore':
        mean = patch_stats['mean']
        std = patch_stats['std']
        return x * std + mean

    return x


# ------------------------------------------------------------------------------
# 1. UTILS & SAFE BOUNDS PER LA TUA FUNZIONE DI PLOT
# ------------------------------------------------------------------------------



def safe_clean(arr):
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().numpy()
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def get_safe_bounds(data, lower_percentile=1, upper_percentile=99):
    vmin = np.percentile(data, lower_percentile)
    vmax = np.percentile(data, upper_percentile)
    if abs(vmax - vmin) < 1e-7:
        vmin = float(np.min(data)) - 1e-5
        vmax = float(np.max(data)) + 1e-5
    return vmin, vmax


def comparison_plots_ok(
    x_real, 
    x_gen, 
    x_lr=None, 
    sources_coords=None,
    title_suffix=""
):
    # 1. Clean & Squeeze inputs
    x_real = safe_clean(x_real)
    x_gen = safe_clean(x_gen)
    if x_lr is not None:
        x_lr = safe_clean(x_lr)

    if x_real.ndim == 5: x_real = x_real[0]
    if x_gen.ndim == 5:  x_gen  = x_gen[0]
    if x_real.ndim == 4: x_real = x_real[0] if x_real.shape[0] in [1, 3] else x_real.squeeze()
    if x_gen.ndim == 4:  x_gen  = x_gen[0] if x_gen.shape[0] in [1, 3] else x_gen.squeeze()

    has_lr = x_lr is not None
    if has_lr:
        if x_lr.ndim == 5: x_lr = x_lr[0]
        if x_lr.ndim == 4: x_lr = x_lr[0] if x_lr.shape[0] in [1, 3] else x_lr.squeeze()

    # 2. Metriche (PSNR e SSIM 3D)
    data_range = x_real.max() - x_real.min()
    if data_range < 1e-7: data_range = 1.0
    
    psnr_val = psnr_metric(x_real, x_gen, data_range=data_range)
    try:
        ssim_val = ssim_metric(x_real, x_gen, data_range=data_range)
    except Exception:
        ssim_val = 0.0

    # 3. Calcolo Proiezioni MIP e Slice Centrali
    mip_real_xy, mip_real_xz, mip_real_yz = np.max(x_real, axis=0), np.max(x_real, axis=1), np.max(x_real, axis=2)
    mip_gen_xy,  mip_gen_xz,  mip_gen_yz  = np.max(x_gen, axis=0),  np.max(x_gen, axis=1),  np.max(x_gen, axis=2)

    cz, cy, cx = x_real.shape[0] // 2, x_real.shape[1] // 2, x_real.shape[2] // 2
    slice_real_xy, slice_real_xz, slice_real_yz = x_real[cz, :, :], x_real[:, cy, :], x_real[:, :, cx]
    slice_gen_xy,  slice_gen_xz,  slice_gen_yz  = x_gen[cz, :, :],  x_gen[:, cy, :],  x_gen[:, :, cx]

    # Mappe di errore (Absolute Difference)
    err_xy = np.abs(mip_real_xy - mip_gen_xy)
    err_xz = np.abs(mip_real_xz - mip_gen_xz)
    err_yz = np.abs(mip_real_yz - mip_gen_yz)

    # 4. Calcolo Bounds per Normalizzazione Colormap
    all_volumes = [
        mip_real_xy.ravel(), mip_real_xz.ravel(), mip_real_yz.ravel(),
        mip_gen_xy.ravel(),  mip_gen_xz.ravel(),  mip_gen_yz.ravel()
    ]
    if has_lr:
        mip_lr_xy = np.max(x_lr, axis=0)
        mip_lr_xz = np.max(x_lr, axis=1)
        mip_lr_yz = np.max(x_lr, axis=2)
        all_volumes.extend([mip_lr_xy.ravel(), mip_lr_xz.ravel(), mip_lr_yz.ravel()])

    # Calcola vmin e vmax basandosi UNICAMENTE sul volume High Resolution (Target)
    mip_real_all = np.concatenate([mip_real_xy.ravel(), mip_real_xz.ravel(), mip_real_yz.ravel()])
    vmin, vmax = get_safe_bounds(mip_real_all, lower_percentile=1, upper_percentile=99)
    

    # 5. Struttura del Plot (5 Righe se c'è LR, altrimenti 4)
    n_rows = 5 if has_lr else 4
    fig = plt.figure(figsize=(16, 3.8 * n_rows))
    gs = fig.add_gridspec(n_rows, 3, height_ratios=[1] * (n_rows - 1) + [0.7])

    main_title = f"Validation SR | PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f} {title_suffix}"
    fig.suptitle(main_title, fontsize=14, fontweight='bold', y=0.995)

    row = 0

    # ROW 1: HIGH RESOLUTION (GT) MIP
    im_gt = fig.add_subplot(gs[row, 0]).imshow(mip_real_xy, origin="lower", cmap="inferno", vmin=vmin, vmax=vmax)
    fig.axes[-1].set_title("MIP XY (Top View)")
    fig.axes[-1].set_ylabel(r"$\bf{HR\ (Target)}$")

    fig.add_subplot(gs[row, 1]).imshow(mip_real_xz, origin="lower", cmap="inferno", vmin=vmin, vmax=vmax)
    fig.axes[-1].set_title("MIP XZ (Front View)")

    fig.add_subplot(gs[row, 2]).imshow(mip_real_yz, origin="lower", cmap="inferno", vmin=vmin, vmax=vmax)
    fig.axes[-1].set_title("MIP YZ (Side View)")
    row += 1

    # ROW (OPZIONALE): LOW RESOLUTION
    if has_lr:
        fig.add_subplot(gs[row, 0]).imshow(mip_lr_xy, origin="lower", cmap="inferno", vmin=vmin, vmax=vmax)
        fig.axes[-1].set_title("MIP XY (Top View)")
        fig.axes[-1].set_ylabel(r"$\bf{Low\ Res}$")

        fig.add_subplot(gs[row, 1]).imshow(mip_lr_xz, origin="lower", cmap="inferno", vmin=vmin, vmax=vmax)
        fig.axes[-1].set_title("MIP XZ (Front View)")

        fig.add_subplot(gs[row, 2]).imshow(mip_lr_yz, origin="lower", cmap="inferno", vmin=vmin, vmax=vmax)
        fig.axes[-1].set_title("MIP YZ (Side View)")
        row += 1

    # ROW 2: SUPER RESOLUTION MIP
    fig.add_subplot(gs[row, 0]).imshow(mip_gen_xy, origin="lower", cmap="inferno", vmin=vmin, vmax=vmax)
    fig.axes[-1].set_title("MIP XY (Top View)")
    fig.axes[-1].set_ylabel(r"$\bf{Super\ Res}$")

    fig.add_subplot(gs[row, 1]).imshow(mip_gen_xz, origin="lower", cmap="inferno", vmin=vmin, vmax=vmax)
    fig.axes[-1].set_title("MIP XZ (Front View)")

    fig.add_subplot(gs[row, 2]).imshow(mip_gen_yz, origin="lower", cmap="inferno", vmin=vmin, vmax=vmax)
    fig.axes[-1].set_title("MIP YZ (Side View)")
    row += 1

    # ROW 3: MAPPA DI ERRORE ASSOLUTO (|HR - SR|)
    im_err = fig.add_subplot(gs[row, 0]).imshow(err_xy, origin="lower", cmap="hot")
    fig.axes[-1].set_title("Error MIP XY")
    fig.axes[-1].set_ylabel(r"$\bf{|HR - SR|}$")

    fig.add_subplot(gs[row, 1]).imshow(err_xz, origin="lower", cmap="hot")
    fig.axes[-1].set_title("Error MIP XZ")

    fig.add_subplot(gs[row, 2]).imshow(err_yz, origin="lower", cmap="hot")
    fig.axes[-1].set_title("Error MIP YZ")
    row += 1

    # ROW 4: PROFILI 1D D'INTENSITÀ (Line Cuts)
    ax_profile = fig.add_subplot(gs[row, :])
    line_gt = x_real[cz, cy, :]
    line_sr = x_gen[cz, cy, :]
    
    ax_profile.plot(line_gt, label='High Resolution (GT)', color='black', linewidth=1.5)
    ax_profile.plot(line_sr, label='Super Resolution (SR)', color='tab:red', linestyle='--', linewidth=1.5)
    
    if has_lr:
        # Se LR ha dimensioni spaziali minori, la ridimensioniamo visivamente sul grafico
        line_lr = x_lr[x_lr.shape[0]//2, x_lr.shape[1]//2, :]
        x_lr_coords = np.linspace(0, len(line_gt) - 1, len(line_lr))
        ax_profile.plot(x_lr_coords, line_lr, label='Low Resolution', color='tab:blue', alpha=0.7, linestyle=':')

    ax_profile.set_title(f"1D Intensity Line Cut (Centered at Z={cz}, Y={cy})", fontsize=11)
    ax_profile.set_xlabel("X (Pixel Index)")
    ax_profile.set_ylabel("Intensità")
    ax_profile.legend(loc='upper right')
    ax_profile.grid(True, linestyle=':', alpha=0.6)

    # Coordinate sorgenti (se presenti)
    if sources_coords is not None and len(sources_coords) > 0:
        xs = [pt[0] for pt in sources_coords]
        ys = [pt[1] for pt in sources_coords]
        for ax in fig.axes[:3]:
            ax.scatter(xs, ys, color='cyan', marker='o', s=40, facecolors='none', linewidths=1.2)

    fig.tight_layout(rect=[0, 0, 0.90, 0.97])

    # Colorbar Principale (Inferno)
    cbar_ax1 = fig.add_axes([0.91, 0.35, 0.015, 0.55])
    cbar1 = fig.colorbar(im_gt, cax=cbar_ax1)
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3, 3))
    cbar1.formatter = formatter
    cbar1.update_ticks()

    # Colorbar Errore (Hot)
    cbar_ax2 = fig.add_axes([0.91, 0.18, 0.015, 0.12])
    cbar2 = fig.colorbar(im_err, cax=cbar_ax2)
    cbar2.formatter = formatter
    cbar2.update_ticks()

    return fig

def draw_img_in_three_dim(img, title: str, output_folder: Path) -> None:
    img = safe_clean(img)
    img = np.squeeze(img)
    
    if img.ndim == 4:
        img = img[0]
    
    if img.ndim != 3:
        print(f"Error: Volume shape is {img.shape}, but must be 3D.")
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
    title: str = "Orthogonal cuts of Data cube", 
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
    axes[0].set_xlabel("Pixels")
    axes[0].set_ylabel("Pixels")
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    im2 = axes[1].imshow(img_recon, cmap=DEFAULT_CMAP, origin='lower', vmin=vmin, vmax=vmax)
    axes[1].set_title("Generated / Reconstructed")
    axes[1].set_xlabel("Pixels")
    axes[1].set_ylabel("Pixels")
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    diff = img_orig - img_recon
    _, v_max_diff = get_safe_bounds(np.abs(diff), lower_percentile=0.0, upper_percentile=99.5)
        
    im3 = axes[2].imshow(diff, cmap="seismic", origin='lower', vmin=-v_max_diff, vmax=v_max_diff)
    axes[2].set_title("Residuals (Orig - Recon)")
    axes[2].set_xlabel("Pixels")
    axes[2].set_ylabel("Pixels")
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
    return fig