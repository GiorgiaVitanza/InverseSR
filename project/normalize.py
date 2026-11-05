import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from glob import glob


from utils.dataset_v3 import normalize_dynamic
from utils.plot_new import comparison_plots_ok



# ------------------------------------------------------------------------------
# 2. TEST DI TUTTE LE NORMALIZZAZIONI SU DIVERSI SUFFISSI DI PATCH
# ------------------------------------------------------------------------------
def test_all_normalizations_multisuffix(patches_dict, json_stats_path, modes=None, output_base_dir="training_plots_preview"):
    if modes is None:
        modes = ['global_robust', 'log_minmax', 'zscore_robust', 'minmax_clip']

    if not os.path.exists(json_stats_path):
        raise FileNotFoundError(f"Impossibile trovare il file JSON: {json_stats_path}")

    with open(json_stats_path, "r") as f:
        stats = json.load(f)

    for patch_label, data_path in patches_dict.items():
        patch_output_dir = os.path.join(output_base_dir, patch_label)
        os.makedirs(patch_output_dir, exist_ok=True)

        raw_data = np.load(data_path).astype(np.float32)
        if raw_data.ndim > 3: 
            raw_data = raw_data.squeeze()

        print(f"\n=======================================================")
        print(f"--- TEST NORM SU TIPO PATCH: [{patch_label}] ---")
        print(f"File: {os.path.basename(data_path)}")
        print(f"RAW -> Min: {raw_data.min():.4e} | Max: {raw_data.max():.4e} | Mean: {raw_data.mean():.4e}")
        print(f"=======================================================")

        for mode in modes:
            norm_cube, _ = normalize_dynamic(raw_data, norm_mode=mode, stats=stats)
            
            if isinstance(norm_cube, torch.Tensor):
                norm_cube = norm_cube.detach().cpu().numpy()

            x_real = norm_cube
            # Simula predizione del modello (con un po' di rumore per distinguere le righe)
            x_gen = norm_cube * 0.9 + np.random.normal(0, 0.01 * norm_cube.std() + 1e-8, norm_cube.shape)

            print(f"Modalità [{mode.upper():<13}] -> Min: {norm_cube.min():.4e} | Max: {norm_cube.max():.4e}")

            fig = comparison_plots_ok(
                x_real=x_real,
                x_gen=x_gen,
                x_lr=None,
                title_suffix=f"[{patch_label}] - MODE: {mode.upper()}"
            )

            save_path = os.path.join(patch_output_dir, f"plot_{patch_label}_{mode}.png")
            fig.savefig(save_path, dpi=200, bbox_inches='tight')
            plt.close(fig)

    print(f"\n✅ Tutti i plot per tutti i tipi di patch sono stati salvati in: '{output_base_dir}/'")


# ------------------------------------------------------------------------------
# 3. ESECUZIONE CON RICERCA AUTOMATICA DEI SUFFISSI
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    BASE_DIR = "/leonardo_scratch/large/userexternal/gvitanza/InverseSR/data/inputs/HI_128x128x128_stride64/train"
    patches_dir = os.path.join(BASE_DIR, "npy_patches")
    
    # Cerca il file delle statistiche nel percorso specifico o nella cartella base
    candidate_json = [
        os.path.join(patches_dir, "dataset_filter_stats.json"),
        os.path.join(BASE_DIR, "dataset_filter_stats.json")
    ]
    json_stats_path = next((p for p in candidate_json if os.path.exists(p)), None)

    # Trova automaticamente un file di esempio per ciascun tipo/suffisso di patch
    all_npy = sorted(glob(os.path.join(patches_dir, "*.npy")))
    
    if len(all_npy) == 0:
        print(f"Nessun file .npy trovato in: {patches_dir}")
    elif json_stats_path is None:
        print(f"File dataset_filter_stats.json non trovato nei percorsi cercati.")
    else:
        # Mappa i vari tipi di patch basandosi sui nomi dei file (es. extract suffix dopo 'patch_XXXXXX_')
        selected_patches = {}
        for p in all_npy:
            fname = os.path.basename(p)
            parts = fname.replace(".npy", "").split("_")
            
            # Estrae la chiave del tipo/suffisso (es. 'MEERKAT_ATCA', 'FAST', ecc.)
            if len(parts) > 2:
                suffix_key = "_".join(parts[2:])
            else:
                suffix_key = "default"

            # Salva la prima patch trovata per quel suffisso
            if suffix_key not in selected_patches:
                selected_patches[suffix_key] = p

        print(f"Tipi di patch identificati nel dataset ({len(selected_patches)}):")
        for k, path in selected_patches.items():
            print(f" - {k}: {os.path.basename(path)}")

        test_all_normalizations_multisuffix(
            patches_dict=selected_patches,
            json_stats_path=json_stats_path,
            output_base_dir="training_plots_preview"
        )