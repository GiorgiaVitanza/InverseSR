import os
import numpy as np
import pandas as pd
from pathlib import Path
from glob import glob

# Lista delle trasformazioni disponibili
AUGMENTATIONS = ["flip_w", "flip_h", "flip_d", "rot90", "rot180", "rot270"]


def transform_data_and_coordinates(
    data: np.ndarray,
    aug_type: str,
    rel_x: float = None,
    rel_y: float = None,
    rel_z: float = None
):
    """
    Applica una trasformazione spaziale 3D a un numpy array (D, H, W)
    e aggiorna coerentemente le coordinate del punto sorgente (x, y, z).
    """
    
    data_3d = data[0]
    D, H, W = data_3d.shape
    new_x, new_y, new_z = rel_x, rel_y, rel_z

    if aug_type == "flip_w":
        # Flip sull'asse W (orizzontale)
        transformed_data = np.flip(data_3d, axis=-1)
        if rel_x is not None:
            new_x = (W - 1) - rel_x

    elif aug_type == "flip_h":
        # Flip sull'asse H (verticale)
        transformed_data = np.flip(data_3d, axis=-2)
        if rel_y is not None:
            new_y = (H - 1) - rel_y

    elif aug_type == "flip_d":
        # Flip sull'asse D (profondità/spettrale)
        transformed_data = np.flip(data_3d, axis=-3)
        if rel_z is not None:
            new_z = (D - 1) - rel_z

    elif aug_type == "rot90":
        # Rotazione 90° antioraria nel piano (H, W)
        transformed_data = np.rot90(data_3d, k=1, axes=(-2, -1))
        if rel_x is not None and rel_y is not None:
            new_x = rel_y
            new_y = (W - 1) - rel_x

    elif aug_type == "rot180":
        # Rotazione 180° nel piano (H, W)
        transformed_data = np.rot90(data_3d, k=2, axes=(-2, -1))
        if rel_x is not None and rel_y is not None:
            new_x = (W - 1) - rel_x
            new_y = (H - 1) - rel_y

    elif aug_type == "rot270":
        # Rotazione 270° nel piano (H, W)
        transformed_data = np.rot90(data_3d, k=3, axes=(-2, -1))
        if rel_x is not None and rel_y is not None:
            new_x = (H - 1) - rel_y
            new_y = rel_x

    else:
        raise ValueError(f"Tipo di augmentation '{aug_type}' non riconosciuto.")


    return transformed_data.copy(), new_x, new_y, new_z


def offline_augment_dataset(
    input_dir: str | Path,
    output_dir: str | Path,
    input_catalog_path: str | Path = None,
    output_catalog_path: str | Path = None,
    num_aug_per_patch: int = 2,
    keep_original: bool = True
):
    """
    Legge tutti i patch da input_dir, genera varianti augmented e le salva in output_dir.
    Se presente un catalogo CSV, aggiorna anche le coordinate rel_x, rel_y, rel_z.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Caricamento catalogo (se presente)
    df_catalog = None
    if input_catalog_path and os.path.exists(input_catalog_path):
        df_catalog = pd.read_csv(input_catalog_path)
        df_catalog.columns = [c.lower().strip() for c in df_catalog.columns]
        if "patch_id" in df_catalog.columns:
            df_catalog = df_catalog.set_index("patch_id")
        else:
            print("Warning: 'patch_id' non presente nel CSV. Il catalogo verrà ignorato.")
            df_catalog = None

    new_catalog_rows = []
    patch_files = sorted(list(input_dir.glob("*.npy")))

    if not patch_files:
        raise FileNotFoundError(f"Nessun file .npy trovato in {input_dir}")

    print(f"Trovati {len(patch_files)} file originali. Inizio data augmentation offline...")

    for file_path in patch_files:
        filename = file_path.name
        data = np.load(file_path).astype(np.float32)

        # Recupero info dal catalogo se disponibili
        has_coords = False
        rel_x, rel_y, rel_z = None, None, None
        row_dict = {}

        if df_catalog is not None and filename in df_catalog.index:
            row_dict = df_catalog.loc[filename].to_dict()
            if all(k in row_dict for k in ("rel_x", "rel_y", "rel_z")):
                rel_x = float(row_dict["rel_x"])
                rel_y = float(row_dict["rel_y"])
                rel_z = float(row_dict["rel_z"])
                has_coords = True

        # A) Salvataggio file ORIGINALE nella nuova cartella
        if keep_original:
            np.save(output_dir / filename, data)
            if df_catalog is not None and filename in df_catalog.index:
                orig_row = row_dict.copy()
                orig_row["patch_id"] = filename
                new_catalog_rows.append(orig_row)

        # B) Generazione e salvataggio delle AUGMENTATION
        chosen_augs = np.random.choice(
            AUGMENTATIONS, 
            size=min(num_aug_per_patch, len(AUGMENTATIONS)), 
            replace=False
        )

        for aug_type in chosen_augs:
            aug_data, new_x, new_y, new_z = transform_data_and_coordinates(
                data, aug_type, rel_x, rel_y, rel_z
            )

            # Nome nuovo file: es. patch_01_rot90.npy
            base_name = file_path.stem
            aug_filename = f"{base_name}_{aug_type}.npy"

            # Salvataggio patch .npy
            np.save(output_dir / aug_filename, aug_data)

            # Aggiornamento riga catalogo per il nuovo file
            if df_catalog is not None and filename in df_catalog.index:
                aug_row = row_dict.copy()
                aug_row["patch_id"] = aug_filename
                if has_coords:
                    aug_row["rel_x"] = new_x
                    aug_row["rel_y"] = new_y
                    aug_row["rel_z"] = new_z
                new_catalog_rows.append(aug_row)

    # 3. Salvataggio del nuovo CSV aggiornato
    if new_catalog_rows and output_catalog_path:
        df_new_catalog = pd.DataFrame(new_catalog_rows)
        df_new_catalog.to_csv(output_catalog_path, index=False)
        print(f"Nuovo catalogo salvato in: {output_catalog_path}")

    print(f"Augmentation completata! Tutti i patch sono in: {output_dir}")


# --- ESEMPIO DI UTILIZZO ---
if __name__ == "__main__":
    offline_augment_dataset(
        input_dir="../../data/inputs/16x128x128_cont_ldev_OK/train/npy_patches",
        output_dir="../../data/inputs/16x128x128_cont_ldev_OK/train_augmented",
        input_catalog_path="../../data/inputs/16x128x128_cont_ldev_OK/train/train_catalogue.csv",
        output_catalog_path="../../data/inputs/16x128x128_cont_ldev_OK/train/train_catalogue_augmented.csv",
        num_aug_per_patch=5,  # Crea 5 varianti casuali per ogni patch
        keep_original=True     # Mantiene anche i file originali
    )