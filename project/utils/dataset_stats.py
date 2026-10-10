import numpy as np


from pathlib import Path
from typing import Union


def compute_dataset_stats(dataset_input: Union[str, Path]):
    input_path = Path(dataset_input)

    # Verifica prima se il percorso esiste
    if not input_path.exists():
        raise FileNotFoundError(
            f"Il percorso specificato non esiste: {dataset_input}"
        )

    # Gestione del singolo file .npy
    if input_path.is_file():
        if input_path.suffix != ".npy":
            raise ValueError(
                f"Il file specificato non ha estensione .npy: {dataset_input}"
            )
        npy_files = [input_path]

    # Gestione della directory
    elif input_path.is_dir():
        npy_files = list(input_path.glob("*.npy"))
        if not npy_files:
            raise FileNotFoundError(
                f"Nessun file .npy trovato nella directory: {dataset_input}"
            )

    else:
        raise ValueError(
            f"Il percorso specificato non è né un file né una directory valida: {dataset_input}"
        )

    print(f"Trovati {len(npy_files)} file .npy. Inizio calcolo statistiche...")

    # Prima passata: Calcolo min, max e percentili globali
    all_data = []
    global_min = float("inf")
    global_max = float("-inf")

    for f in npy_files:
        data = np.load(f)
        global_min = min(global_min, float(np.min(data)))
        global_max = max(global_max, float(np.max(data)))
        all_data.append(data.ravel())

    # Concateniamo i dati per calcoli distribuzionali globali
    concatenated_data = np.concatenate(all_data)

    p5 = float(np.percentile(concatenated_data, 5))
    p95 = float(np.percentile(concatenated_data, 95))
    mean = float(np.mean(concatenated_data))
    std = float(np.std(concatenated_data))

    stats = {
        "min": global_min,
        "max": global_max,
        "p5": p5,
        "p95": p95,
        "min_arcsinh": float(np.arcsinh(global_min)),
        "max_arcsinh": float(np.arcsinh(global_max)),
        "p5_arcsinh": float(np.arcsinh(p5)),
        "p95_arcsinh": float(np.arcsinh(p95)),
        "mean": mean,
        "std": std,
    }

    return stats


if __name__ == "__main__":
    
    data_dir = "../../data/inputs/16x128x128_cont_ldev_OK/test_augmented"
    results = compute_dataset_stats(data_dir)

    print("\n--- Risultato per argparse ---")
    print("default=" + repr(results))