from typing import List, Optional
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from utils.const import FITS_LIMIT, FITS_MEAN, FITS_STD




def denormalize_data(x, norm_mode):
    """Denormalizza in base alla modalità scelta per tornare ai Jy/beam"""
   
    
    mode = norm_mode

    if mode == 'global_sym':
        print("Denormalizzazione globale simmetrica")
        # Inverti: x_norm = (x_scaled + 1) / 2 -> x_scaled = x_norm * 2 - 1
        x_phys = (x * 2.0 - 1.0) * FITS_LIMIT
        return x_phys
        
    elif mode == 'local':
        # La denormalizzazione locale accurata è impossibile senza salvare p_min/p_max per ogni patch.
        # Come fallback, usiamo i globali, ma i valori saranno approssimativi.
        print("Denormalizzazione locale: usando valori globali come approssimazione")
        v_min, v_max = -1.47e-03, 1.52e-03
        return x * (v_max - v_min) + v_min
        
    elif mode == 'zscore':
        print("denormalizzazione z-score")
        # Inverti: x_norm = (data - FITS_MEAN) / FITS_STD
        return x * FITS_STD + FITS_MEAN
        
    return x

def comparison_plots_ok(x, x_hat, flag = 'test'):
                    try:                        
                        img_orig = x[0, 0].detach().cpu().numpy()      # Cubo originale (128, 128, 128)
                        img_recon = x_hat[0, 0].detach().cpu().numpy() # Cubo ricostruito
                    except:
                        img_orig = x
                        img_recon = x_hat

                    # 1. Calcoliamo la Slice Centrale
                    mid_z = img_orig.shape[0] // 2
                    mid_x = img_orig.shape[1] // 2
                    mid_y = img_orig.shape[2] // 2
                    slice_orig_z = img_orig[mid_z]
                    slice_recon_z = img_recon[mid_z]
                    slice_orig_x = img_orig[:, mid_x, :]
                    slice_recon_x = img_recon[:, mid_x, :]
                    slice_orig_y = img_orig[:, :, mid_y]
                    slice_recon_y = img_recon[:, :, mid_y]

                    # 2. Calcoliamo il MOMENTO 0 (Somma lungo Z)
                    # Questo fa emergere la galassia anche se è debole
                    mom0_orig = np.sum(img_orig, axis=0)
                    mom0_recon = np.sum(img_recon, axis=0)

                    # Creiamo una griglia 2x2 per il confronto
                    fig, axes = plt.subplots(4, 2, figsize=(12, 10))
                    
                    
                    """ # Calcoliamo un vmax comune per le slice per vedere la differenza di contrasto
                    vmax_slice_z = np.percentile(slice_orig_z, 99.9)
                    vmax_slice_y = np.percentile(slice_orig_y, 99.9)
                    vmax slice_x = np.percentile(slice_orig_x, 99.9)"""
                    # Posizione 1-1: Slice Z originale
                    im1 = axes[0, 0].imshow(slice_orig_z, cmap='hot', origin='lower')
                    axes[0, 0].set_title(f"Originale (Slice Z={mid_z})")
                    plt.colorbar(im1, ax=axes[0, 0])
                    plt.subplots_adjust(hspace=0.8)
                    # Posizione 2-1: Slice X originale (usiamo lo stesso vmax per coerenza)
                    im2 = axes[1, 0].imshow(slice_orig_x, cmap='hot', origin='lower')
                    axes[1, 0].set_title(f"Originale (Slice X={mid_x})")
                    plt.colorbar(im2, ax=axes[1, 0])
                    plt.subplots_adjust(hspace=0.8)
                    # Posizione 3-1: Slice Y originale (usiamo lo stesso vmax per coerenza)
                    im3 = axes[2, 0].imshow(slice_orig_y, cmap='hot', origin='lower')
                    axes[2, 0].set_title(f"Originale (Slice Y={mid_y})")
                    plt.colorbar(im3, ax=axes[2, 0])
                    plt.subplots_adjust(hspace=0.8)
                    # --- MOMENTO 0 ---
                    # Calcoliamo un vmax comune per le proiezioni
                    # vmax_mom = np.percentile(mom0_orig, 99.9)

                    im4 = axes[3, 0].imshow(mom0_orig, cmap='hot', origin='lower')
                    axes[3, 0].set_title("Originale (Momento 0)")
                    plt.colorbar(im4, ax=axes[3, 0])

                
                    im5 = axes[0, 1].imshow(slice_recon_z, cmap='hot', origin='lower')
                    axes[0, 1].set_title("Ricostruito (Slice Z)")
                    plt.colorbar(im5, ax=axes[0, 1])
                    plt.subplots_adjust(hspace=0.8)
                    im6 = axes[1, 1].imshow(slice_recon_x, cmap='hot', origin='lower')
                    axes[1, 1].set_title("Ricostruito (Slice X)")
                    plt.colorbar(im6, ax=axes[1, 1])
                    plt.subplots_adjust(hspace=0.8)
                    im7 = axes[2, 1].imshow(slice_recon_y, cmap='hot', origin='lower')
                    axes[2, 1].set_title("Ricostruito (Slice Y)")
                    plt.colorbar(im7, ax=axes[2, 1])
                    plt.subplots_adjust(hspace=0.8)
                    im8 = axes[3, 1].imshow(mom0_recon, cmap='hot', origin='lower')
                    axes[3, 1].set_title("Ricostruito (Momento 0)")
                    plt.colorbar(im8, ax=axes[3, 1])
                    

                    plt.tight_layout()
                        
                    return fig

def draw_img_in_three_dim(img, title: str, output_folder: Path) -> None:
    """
    Adattato per Datacube Astronomici.
    Gestisce input sia NumPy che Torch, rimuovendo dimensioni extra.
    """
    # 1. Conversione in NumPy se è un Tensor
    if hasattr(img, "detach"):
        img = img.detach().cpu().numpy()
    
    # 2. Pulizia dimensioni (Squeeze)
    img = np.squeeze(img)
    
    # 3. Gestione caso multi-canale (es. i 3 canali visti prima)
    if img.ndim == 4:
        # Se abbiamo [C, D, H, W], prendiamo il primo canale
        img = img[0]
    
    if img.ndim != 3:
        print(f"Errore: Il volume ha shape {img.shape}, ma deve essere 3D.")
        return

    si, sj, sk = img.shape
    # Nomi più appropriati per un Datacube (RA, Dec, Freq/Vel)
    # Di solito: Axial -> RA/Dec, Sagittal/Coronal -> Piani con Frequenza
    dim_names = ["RA-Dec", "Freq-Dec", "Freq-RA"]
    
    fig, ax = plt.subplots()
    
    # --- PIANO 1: XY (Axial / RA-Dec) ---
    img_slice = np.rot90(img[si//2, :, :], 1)
    ax.imshow(img_slice, cmap="hot", origin='lower') 
    ax.axis("off")
    ax.set_title(f"{dim_names[0]} (slice {sk // 2})")
    fig.savefig(output_folder / f"{title}_{dim_names[0]}.png", 
                bbox_inches="tight", pad_inches=0.1, dpi=300)

    # --- PIANO 2: XZ (Sagittal / Freq-Dec) ---
    img_slice = np.rot90(img[:, sj // 2, :], 1)
    ax.imshow(img_slice, cmap="hot", origin='lower')
    ax.axis("off")
    ax.set_title(f"{dim_names[1]} (slice {sj // 2})")
    fig.savefig(output_folder / f"{title}_{dim_names[1]}.png", 
                bbox_inches="tight", pad_inches=0.1, dpi=300)

    # --- PIANO 3: YZ (Coronal / Freq-RA) ---
    img_slice = np.rot90(img[:, :, sk//2], 1)
    ax.imshow(img_slice, cmap="hot", origin='lower')
    ax.axis("off")
    ax.set_title(f"{dim_names[2]} (slice {si // 2})")
    fig.savefig(output_folder / f"{title}_{dim_names[2]}.png", 
                bbox_inches="tight", pad_inches=0.1, dpi=300)

    plt.close(fig)
    print(f"Salvate 3 proiezioni ortogonali in: {output_folder}")


# Configurazione default per Astro
DEFAULT_CMAP = "hot"  # O 'viridis', 'magma', 'cividis'
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
            ax.imshow(imgs_list[idx * ncols + jdx], cmap=DEFAULT_CMAP)
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
    si, sj, sk = img.shape # Freq Ra Dec 
    img_slice = np.rot90(img[si // 2, :, :], -1)
    img = ax.imshow(img_slice, cmap=DEFAULT_CMAP)
    
    ax.set_title(title)
    ax.set_xlabel("Pixels")
    ax.set_ylabel("Pixels")
    plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    
    fig.savefig(
        output_folder / f"{step}_{title}.png",
        bbox_inches="tight",
        pad_inches=0,
        format="png",
        dpi=300,
    )
    # close
    plt.close(fig)


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

    # 4. Taglio Ra-Dec (Slice centrale lungo Z):
    #    Questo mostra la distribuzione spaziale a una frequenza/velocità specific
    img_ra_dec = cube[nz // 2, :, :]

    imgs = [img_spatial, img_spectral_ra, img_spectral_dec, img_ra_dec]
    titles = ["Spatial (Moment 0)", "Spectral (Z - RA)", "Spectral (Z - Dec)", "RA-Dec (Z - Center)"]
    
    # --- Plotting ---
    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(2, 2)
    
    for i, img in enumerate(imgs):
        ax = plt.subplot(gs[i])
        
        # origin='lower' è CRUCIALE per i FITS, altrimenti l'immagine è capovolta
        im = ax.imshow(img, cmap=DEFAULT_CMAP, origin='lower')
        
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
