import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
from astropy.io import fits
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def create_3d_slice_gif(patch_dir, output_gif_dir, num_samples=3):
    """Crea GIF animate con rotazione 3D del cubo e piani di taglio lungo l'asse Z."""
    os.makedirs(output_gif_dir, exist_ok=True)
    
    all_files = [f for f in os.listdir(patch_dir) if f.endswith('.npy')]
    if not all_files:
        print(f"❌ Nessun file .npy trovato in: {patch_dir}")
        return

    sample_patches = np.random.choice(
        all_files, 
        size=min(num_samples, len(all_files)), 
        replace=False
    )
    
    for patch_id in sample_patches:
        file_path = os.path.join(patch_dir, patch_id)

        data = np.load(file_path)
        if data.ndim == 4:
            data = data[0]

        depth_Z, height_Y, width_X = data.shape
        vmax = np.percentile(data, 99.8) if np.max(data) > 0 else 1.0
        vmin = 0.0

        fig = plt.figure(figsize=(9, 8))
        ax = fig.add_subplot(111, projection='3d')

        X_grid, Y_grid = np.meshgrid(np.arange(width_X), np.arange(height_Y))

        def update(frame):
            ax.clear()
            z_slice_idx = frame
            slice_2d = data[z_slice_idx, :, :]

            ax.plot_surface(
                X_grid, 
                Y_grid, 
                np.full_like(X_grid, z_slice_idx),
                rstride=2, cstride=2,
                facecolors=plt.cm.inferno((slice_2d - vmin) / (vmax - vmin + 1e-8)),
                shade=False,
                alpha=0.85
            )

            ax.set_xlim(0, width_X)
            ax.set_ylim(0, height_Y)
            ax.set_zlim(0, depth_Z)

            ax.set_xlabel("X (Pixel)")
            ax.set_ylabel("Y (Pixel)")
            ax.set_zlabel("Z (Canali / Profondità)")
            
            angle = (frame / depth_Z) * 360
            ax.view_init(elev=25, azim=angle)
            ax.set_title(f"Patch 3D: {patch_id}\nScansione Z = {z_slice_idx}/{depth_Z-1}")

        ani = animation.FuncAnimation(
            fig, update, frames=range(depth_Z), interval=150
        )
        
        gif_filename = f"3d_cube_{os.path.splitext(patch_id)[0]}.gif"
        save_path = os.path.join(output_gif_dir, gif_filename)
        
        ani.save(save_path, writer='pillow', fps=6)
        plt.close(fig)
        print(f"✅ GIF 3D salvata: {save_path}")


def process_radio_blind_split(
    fits_path,
    output_dir,
    patch_size=(128, 128, 128),
    stride=128,
    splits=(0.7, 0.15, 0.15),  # Ratio per (Train, Val, Test)
    seed=42,
    data_name="",
    output_format="npy",
    skip_empty=False,
    empty_threshold=1e-6
):
    """Estrae i patch 3D in modalità blind e li suddivide in train, val e test set."""
    assert sum(splits) == 1.0, "La somma delle frazioni di split deve fare 1.0"

    print(f"\nApertura FITS: {fits_path}")
    hdul = fits.open(fits_path, memmap=True, mode="readonly", ignore_missing_simple=True)
    raw_data_ref = hdul[0].data
    shape = raw_data_ref.shape
    
    Z, Y, X = shape[1:] if len(shape) == 4 else shape
    print(f"Dimensioni Cubo: Z={Z}, Y={Y}, X={X}")

    # 1. Generazione di tutti i ritagli in memoria e selezione coordinate
    all_patches = []
    patch_count = 0
    skipped_count = 0

    print("Scansione cubo 3D e ritaglio in corso...")
    for z in tqdm(range(0, Z - patch_size[0] + 1, stride), desc="Z-axis"):
        for y in range(0, Y - patch_size[1] + 1, stride):
            for x in range(0, X - patch_size[2] + 1, stride):

                if len(raw_data_ref.shape) == 4:
                    d_slice = raw_data_ref[0, z:z+patch_size[0], y:y+patch_size[1], x:x+patch_size[2]]
                else:
                    d_slice = raw_data_ref[z:z+patch_size[0], y:y+patch_size[1], x:x+patch_size[2]]

                p_data = np.nan_to_num(np.array(d_slice, dtype=np.float32), nan=0.0)

                if skip_empty and np.max(np.abs(p_data)) < empty_threshold:
                    skipped_count += 1
                    continue

                base_name = f"patch_{patch_count:06d}"
                all_patches.append({
                    "base_name": base_name,
                    "data": p_data,
                    "x": x, "y": y, "z": z,
                    "max_intensity": float(np.max(p_data)),
                    "mean_intensity": float(np.mean(p_data))
                })
                patch_count += 1

    hdul.close()
    print(f"Estratti {len(all_patches)} patch validi. Scartati {skipped_count} patch vuoti.")

    # 2. Calcolo degli Split (Train / Val / Test)
    train_ratio, val_ratio, test_ratio = splits
    
    # Primo split: dividi Train da (Val + Test)
    train_data, val_test_data = train_test_split(
        all_patches, train_size=train_ratio, random_state=seed, shuffle=True
    )
    
    # Secondo split: dividi Val e Test proporzionalmente
    relative_val_ratio = val_ratio / (val_ratio + test_ratio)
    # Controlla quanti elementi ci sono prima dello split
    if len(val_test_data) == 1:
        val_data = val_test_data
        test_data = []
    else:
        val_data, test_data = train_test_split(
            val_test_data, train_size=relative_val_ratio, random_state=seed, shuffle=True
        )

    split_datasets = {
        "train": train_data,
        "val": val_data,
        "test": test_data
    }

    # 3. Salvataggio su disco organizzato per sotto-cartelle
    for subset_name, subset_patches in split_datasets.items():
        subset_dir = os.path.join(output_dir, subset_name)
        
        npy_dir = os.path.join(subset_dir, "npy_patches")
        fits_dir = os.path.join(subset_dir, "fits_patches")
        
        if output_format in ["npy", "both"]: os.makedirs(npy_dir, exist_ok=True)
        if output_format in ["fits", "both"]: os.makedirs(fits_dir, exist_ok=True)

        records = []
        for item in subset_patches:
            base_name = item["base_name"]
            p_data = item["data"]
            x, y, z = item["x"], item["y"], item["z"]

            # Salvataggio .npy
            if output_format in ["npy", "both"]:
                np.save(os.path.join(npy_dir, f"{base_name}_{data_name}.npy"), p_data[np.newaxis, ...])

            # Salvataggio .fits
            if output_format in ["fits", "both"]:
                patch_header = fits.Header()
                patch_header["NAXIS1"], patch_header["NAXIS2"], patch_header["NAXIS3"] = patch_size[2], patch_size[1], patch_size[0]
                fits.writeto(os.path.join(fits_dir, f"{base_name}.fits"), p_data, patch_header, overwrite=True)

            records.append({
                "patch_id": f"{base_name}.npy",
                "global_x_min": x, "global_x_max": x + patch_size[2],
                "global_y_min": y, "global_y_max": y + patch_size[1],
                "global_z_min": z, "global_z_max": z + patch_size[0],
                "max_intensity": item["max_intensity"],
                "mean_intensity": item["mean_intensity"],
            })

        # Salva indice CSV per ogni sotto-insieme
        pd.DataFrame(records).to_csv(os.path.join(subset_dir, f"{subset_name}_index.csv"), index=False)
        print(f" Subset [{subset_name.upper()}]: Salvati {len(subset_patches)} patch in {subset_dir}")


if __name__ == "__main__":
    FITS_PATH = "/leonardo_scratch/large/userexternal/gvitanza/MeerKATFornaxSurvey/t06_1kms_NGC1436_image_mos.fits"
    BASE_OUT_DIR = "/leonardo_scratch/large/userexternal/gvitanza/InverseSR/data/inputs/MeerkatNGC1436"

    # 1. Generazione e Split dei Patch
    process_radio_blind_split(
        fits_path=FITS_PATH,
        output_dir=BASE_OUT_DIR,
        patch_size=(16, 128, 128),
        stride=128,
        splits=(0.8, 0.1, 0.1),  # 70% Train, 15% Validation, 15% Test
        seed=42,
        data_name="meerkatNGC1436",
        output_format="npy",
        skip_empty=False
    )

    # 2. Generazione GIF animate di verifica dal subset Train
    print("\n--- Generazione GIF di Verifica (Train Set) ---")
    NPY_TRAIN_DIR = os.path.join(BASE_OUT_DIR, "train", "npy_patches")
    PLOT_GIF_DIR = os.path.join(BASE_OUT_DIR, "gifs_verification")
    
    create_3d_slice_gif(
        patch_dir=NPY_TRAIN_DIR,
        output_gif_dir=PLOT_GIF_DIR,
        num_samples=3
    )