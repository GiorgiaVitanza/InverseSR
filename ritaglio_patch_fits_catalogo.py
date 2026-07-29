import os
import matplotlib
# 1. Imposta il backend Non-Interattivo PRIMA di importare pyplot (Per HPC Leonardo)
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def create_3d_slice_gif(patch_dir, catalog_path, output_gif_dir, num_samples=3):
    """Crea GIF animate che scorrono l'asse Z di un patch, mostrando le sorgenti attive in ogni slice."""
    os.makedirs(output_gif_dir, exist_ok=True)
    if not os.path.exists(catalog_path):
        print(f"❌ Catalogo non trovato: {catalog_path}")
        return

    df_cat = pd.read_csv(catalog_path)
    
    unique_patches = df_cat['patch_id'].unique()
    if len(unique_patches) == 0:
        print("❌ Nessun patch trovato nel catalogo.")
        return

    sample_patches = np.random.choice(unique_patches, size=min(num_samples, len(unique_patches)), replace=False)
    
    for patch_id in sample_patches:
        file_path = os.path.join(patch_dir, patch_id)
        if not os.path.exists(file_path):
            print(f"⚠️ Patch non trovato sul disco: {file_path}")
            continue

        # Caricamento patch .npy (C, D, H, W) -> (Z, Y, X)
        data = np.load(file_path)
        if data.ndim == 4:
            data = data[0]

        depth_Z, height_Y, width_X = data.shape
        patch_sources = df_cat[df_cat['patch_id'] == patch_id]

        fig, ax = plt.subplots(figsize=(7, 6))
        vmax = np.percentile(data, 99.5) if np.max(data) > 0 else 1.0

        def update(z):
            ax.clear()
            slice_2d = data[z, :, :]
            
            im = ax.imshow(slice_2d, origin="lower", cmap="inferno", vmax=vmax, vmin=0)
            
            # Filtra le sorgenti attive in questo canale Z (+/- 1 slice)
            active_sources = patch_sources[np.abs(patch_sources['rel_z'] - z) <= 1.0]
            
            if not active_sources.empty:
                ax.scatter(
                    active_sources['rel_x'],
                    active_sources['rel_y'],
                    s=100,
                    edgecolors="cyan",
                    facecolors="none",
                    linewidths=2,
                    label=f"Attive in Z={z}"
                )
            
            ax.set_title(f"Patch: {patch_id}\nSlice Z = {z}/{depth_Z-1} | Sorgenti attive: {len(active_sources)}")
            ax.set_xlabel("X (Pixel Relativi)")
            ax.set_ylabel("Y (Pixel Relativi)")
            ax.set_xlim(0, width_X)
            ax.set_ylim(0, height_Y)
            if not active_sources.empty:
                ax.legend(loc="upper right")

        ani = animation.FuncAnimation(fig, update, frames=range(depth_Z), interval=333)
        
        gif_filename = f"animation_{os.path.splitext(patch_id)[0]}.gif"
        save_path = os.path.join(output_gif_dir, gif_filename)
        
        ani.save(save_path, writer='pillow', fps=3)
        plt.close(fig)
        print(f"✅ GIF salvata con successo: {save_path}")


def split_original_catalog(catalog_path, train_ratio=0.8, seed=42):
    df_cat = pd.read_table(catalog_path, sep=r"\s+")
    train_df, test_df = train_test_split(
        df_cat, train_size=train_ratio, random_state=seed, shuffle=True
    )
    print(f"Split completato: {len(train_df)} train, {len(test_df)} test")
    return train_df, test_df


def process_radio_multiformat(
    fits_path,
    df_cat,
    output_dir,
    subset_name="train",
    patch_size=(128, 128, 128),
    stride=128,
    output_format="npy",
):
    print(f"\nApertura FITS: {fits_path}")
    hdul = fits.open(fits_path, memmap=True, mode="readonly")
    header_originale = hdul[0].header
    wcs = WCS(header_originale)
    if wcs.naxis == 4:
        wcs = wcs.dropaxis(3)

    raw_data_ref = hdul[0].data
    shape = raw_data_ref.shape
    Z, Y, X = shape[1:] if len(shape) == 4 else shape

    print(f"Caricamento catalogo: {len(df_cat)} sorgenti")
    df_cat = df_cat.copy()

    col_freq = "central_freq"
    col_flux = "line_flux_integral"

    sky_coords = df_cat[["ra", "dec", "central_freq"]].values
    pixels = wcs.all_world2pix(sky_coords, 0)

    df_cat["x_pix"] = pixels[:, 0]
    df_cat["y_pix"] = pixels[:, 1]
    df_cat["z_pix"] = pixels[:, 2]

    subset_dir = os.path.join(output_dir, subset_name)
    paths = {
        fmt: os.path.join(subset_dir, f"{fmt}_patches")
        for fmt in (
            ["fits", "npy"] if output_format == "both" else [output_format]
        )
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)

    master_records = []
    patch_count = 0

    for z in tqdm(range(0, Z - patch_size[0] + 1, stride), desc="Z-axis"):
        cat_z = df_cat[
            (df_cat["z_pix"] >= z) & (df_cat["z_pix"] < z + patch_size[0])
        ]
        if cat_z.empty:
            continue

        for y in range(0, Y - patch_size[1] + 1, stride):
            for x in range(0, X - patch_size[2] + 1, stride):

                sources = cat_z[
                    (cat_z["y_pix"] >= y)
                    & (cat_z["y_pix"] < y + patch_size[1])
                    & (cat_z["x_pix"] >= x)
                    & (cat_z["x_pix"] < x + patch_size[2])
                ].copy()

                if not sources.empty:
                    base_name = f"patch_{patch_count:06d}"

                    # 1. Estrazione del ritaglio dati dal FITS
                    if len(raw_data_ref.shape) == 4:
                        d_slice = raw_data_ref[
                            0,
                            z : z + patch_size[0],
                            y : y + patch_size[1],
                            x : x + patch_size[2],
                        ]
                    else:
                        d_slice = raw_data_ref[
                            z : z + patch_size[0],
                            y : y + patch_size[1],
                            x : x + patch_size[2],
                        ]

                    p_data = np.nan_to_num(
                        np.array(d_slice, dtype=np.float32), nan=0.0
                    )

                    # 2. SALVATAGGIO FISICO DEI FILE (.npy / .fits)
                    if "fits" in paths:
                        patch_header = header_originale.copy()
                        patch_header["NAXIS1"] = patch_size[2]
                        patch_header["NAXIS2"] = patch_size[1]
                        patch_header["NAXIS3"] = patch_size[0]
                        patch_header["CRPIX1"] -= x
                        patch_header["CRPIX2"] -= y
                        patch_header["CRPIX3"] -= z
                        fits.writeto(
                            os.path.join(paths["fits"], f"{base_name}.fits"),
                            p_data,
                            patch_header,
                            overwrite=True,
                        )
                    if "npy" in paths:
                        np.save(
                            os.path.join(paths["npy"], f"{base_name}.npy"),
                            p_data[np.newaxis, ...],
                        )

                    # 3. SALVATAGGIO DI TUTTE LE SORGENTI NEL CATALOGO CSV
                    for _, src in sources.iterrows():
                        record = {
                            "patch_id": f"{base_name}.npy",
                            "n_sources_in_patch": len(sources),
                            "source_id": src["id"],
                            "rel_x": src["x_pix"] - x,
                            "rel_y": src["y_pix"] - y,
                            "rel_z": src["z_pix"] - z,
                            "line_flux_integral": src[col_flux],
                            "hi_size": src["hi_size"],
                            "w20": src["w20"],
                            "central_freq": src[col_freq],
                            "i": src["i"],
                        }
                        master_records.append(record)

                    patch_count += 1

    hdul.close()
    master_df = pd.DataFrame(master_records)
    
    # Salva il file CSV con il nome corretto corrispondente a subset_name
    catalog_filename = f"{subset_name}_catalog.csv"
    master_df.to_csv(
        os.path.join(subset_dir, catalog_filename), index=False
    )
    return output_dir, master_df


if __name__ == "__main__":
    FITS_PATH = "./data/inputs/sky_dev_v2.fits"
    CATALOG_PATH = "./data/inputs/sky_dev_truthcat_v2.txt"
    BASE_OUT_DIR = "./data/inputs/128x128x128_stride128_sky_dev"

    # 1. Split delle sorgenti
    train_cat, test_cat = split_original_catalog(
        CATALOG_PATH, train_ratio=0.8
    )

    # 2. Processamento Train Set
    print("\n--- Processing TRAIN SET ---")
    process_radio_multiformat(
        fits_path=FITS_PATH,
        df_cat=train_cat,
        output_dir=BASE_OUT_DIR,
        subset_name="train",
        patch_size=(128, 128, 128),
        stride=128,
        output_format="npy",
    )

    # 3. Processamento Test Set
    print("\n--- Processing TEST SET ---")
    process_radio_multiformat(
        fits_path=FITS_PATH,
        df_cat=test_cat,
        output_dir=BASE_OUT_DIR,
        subset_name="test",
        patch_size=(128, 128, 128),
        stride=128,
        output_format="npy",
    )

    # 4. Generazione GIF animate di verifica
    print("\n--- Generazione GIF di Verifica ---")
    NPY_TRAIN_DIR = os.path.join(BASE_OUT_DIR, "train", "npy_patches")
    CATALOG_TRAIN_PATH = os.path.join(
        BASE_OUT_DIR, "train", "train_catalog.csv"
    )
    PLOT_GIF_DIR = os.path.join(BASE_OUT_DIR, "gifs_verification")
    
    create_3d_slice_gif(
        patch_dir=NPY_TRAIN_DIR,
        catalog_path=CATALOG_TRAIN_PATH,
        output_gif_dir=PLOT_GIF_DIR,
        num_samples=3
    )