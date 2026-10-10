import numpy as np
import pandas as pd
from pathlib import Path

# Definizione di tutte le 7 trasformazioni uniche e non banali nel piano (H, W),
# eventualmente estese con il flip sul piano di profondità D (flip_z).
# Ciascuna tupla rappresenta: (k_rot_90, flip_h_flag, flip_w_flag, flip_d_flag)
UNIQUE_TRANSFORMS = {
    "rot90":       (1, False, False, False),
    "rot180":      (2, False, False, False),
    "rot270":      (3, False, False, False),
    "flip_w":      (0, False, True,  False),
    "flip_h":      (0, True,  False, False),
    "flip_d":      (0, False, False, True),
    "rot90_flip":  (1, False, True,  False),
    "rot180_flip": (2, False, True,  False),
}


def transform_data_and_coordinates(
    data: np.ndarray,
    aug_type: str,
    rel_x: float = None,
    rel_y: float = None,
    rel_z: float = None
):
    """
    Applica una trasformazione spaziale 3D deterministica a un numpy array (1, D, H, W)
    oppure (D, H, W) e aggiorna coerentemente le coordinate del punto (x, y, z).
    """
    # Gestione sia di array 4D (1, D, H, W) che 3D (D, H, W)
    has_channel_dim = (data.ndim == 4)
    data_3d = data[0] if has_channel_dim else data

    D, H, W = data_3d.shape
    new_x, new_y, new_z = rel_x, rel_y, rel_z
    transformed = data_3d.copy()

    k_rot, do_flip_h, do_flip_w, do_flip_d = UNIQUE_TRANSFORMS[aug_type]

    # 1. Flip sul piano D (Profondità)
    if do_flip_d:
        transformed = np.flip(transformed, axis=0)
        if new_z is not None:
            new_z = (D - 1) - new_z

    # 2. Flip Verticale (H)
    if do_flip_h:
        transformed = np.flip(transformed, axis=1)
        if new_y is not None:
            new_y = (H - 1) - new_y

    # 3. Flip Orizzontale (W)
    if do_flip_w:
        transformed = np.flip(transformed, axis=2)
        if new_x is not None:
            new_x = (W - 1) - new_x

    # 4. Rotazione nel piano (H, W) di k * 90 gradi in senso antiorario
    if k_rot > 0:
        transformed = np.rot90(transformed, k=k_rot, axes=(1, 2))
        if new_x is not None and new_y is not None:
            curr_H, curr_W = H, W
            # Se abbiamo applicato le flippate precedenti, H e W sono invariati,
            # ma durante le rotazioni ad angolo retto le dimensioni W ed H si scambiano.
            for _ in range(k_rot):
                # Rotazione antioraria 90°: (x, y) -> (y, W_corrente - 1 - x)
                tmp_x = new_y
                tmp_y = (curr_W - 1) - new_x
                new_x, new_y = tmp_x, tmp_y
                curr_H, curr_W = curr_W, curr_H  # Scambio dimensioni per il passo successivo

    if has_channel_dim:
        transformed = np.expand_dims(transformed, axis=0)

    return transformed.copy(), new_x, new_y, new_z


def offline_augment_dataset(
    input_dir: str | Path,
    output_dir: str | Path,
    num_aug_per_patch: int = 2,
    keep_original: bool = True,
    seed: int = 42
):
    """
    Legge tutte le patch da input_dir, applica trasformazioni uniche e non ridondanti
    e le salva in output_dir aggiornando il catalogo CSV.
    """
    np.random.seed(seed)
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    patch_files = sorted(list(input_dir.glob("*.npy")))

    if not patch_files:
        raise FileNotFoundError(f"Nessun file .npy trovato in {input_dir}")

    # Limita il numero massimo di augmentation uniche disponibili
    available_aug_keys = list(UNIQUE_TRANSFORMS.keys())
    max_augs = len(available_aug_keys)
    effective_num_aug = min(num_aug_per_patch, max_augs)

    print(f"Trovati {len(patch_files)} file originali. "
          f"Generazione di {effective_num_aug} varianti uniche per patch...")

    for file_path in patch_files:
        filename = file_path.name
        data = np.load(file_path).astype(np.float32)

        has_coords = False
        rel_x, rel_y, rel_z = None, None, None
        row_dict = {}


        # B) Selezione casuale MA senza ripetizioni tra le trasformazioni geometricamente uniche
        chosen_augs = np.random.choice(
            available_aug_keys,
            size=effective_num_aug,
            replace=False
        )

        for aug_type in chosen_augs:
            aug_data, new_x, new_y, new_z = transform_data_and_coordinates(
                data, aug_type, rel_x, rel_y, rel_z
            )

            base_name = file_path.stem
            aug_filename = f"{base_name}_{aug_type}.npy"

            # Salvataggio patch
            np.save(output_dir / aug_filename, aug_data)

            
    print(f"Augmentation completata con successo! File salvati in: {output_dir}")


if __name__ == "__main__":
    offline_augment_dataset(
        input_dir="../../data/inputs/16x128x128_cont_ldev_OK/test/npy_patches",
        output_dir="../../data/inputs/16x128x128_cont_ldev_OK/test_augmented",
        num_aug_per_patch=5,
        keep_original=True,
        seed=42
    )