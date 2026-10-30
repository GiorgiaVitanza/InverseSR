import json
import os
from pathlib import Path
from typing import Union
import numpy as np
from tqdm import tqdm


def compute_dataset_stats(dataset_input: Union[str, Path], subsample_stride: int = 10):
    input_path = Path(dataset_input)

    if not input_path.exists():
        raise FileNotFoundError(f"Il percorso specificato non esiste: {dataset_input}")

    if input_path.is_file():
        if input_path.suffix != ".npy":
            raise ValueError(f"Il file specificato non ha estensione .npy: {dataset_input}")
        npy_files = [input_path]
    elif input_path.is_dir():
        npy_files = list(input_path.glob("*.npy"))
        if not npy_files:
            raise FileNotFoundError(f"Nessun file .npy trovato nella directory: {dataset_input}")
    else:
        raise ValueError(f"Percorso non valido: {dataset_input}")

    print(f"Trovati {len(npy_files)} file .npy. Inizio calcolo statistiche in Streaming/Subsampling...")

    sampled_values = []

    # Calcolo streaming della media e std per evitare imperfezioni col subsampling
    total_sum = 0.0
    total_sq_sum = 0.0
    total_count = 0

    for f in tqdm(npy_files, desc="Elaborazione file .npy"):
        data = np.load(f).astype(np.float32)
        flat_data = data.ravel()

        # Aggiornamento parametri per Media e Deviazione Standard esatte
        total_sum += float(np.sum(flat_data))
        total_sq_sum += float(np.sum(flat_data**2))
        total_count += flat_data.size

        # Sottocampionamento uniforme per il calcolo dei percentili (risparmia RAM)
        sampled_values.append(flat_data[::subsample_stride])

    print("Concatenazione dei campioni per il calcolo dei percentili...")
    concatenated_samples = np.concatenate(sampled_values)

    # Media e Std esatte
    mean = total_sum / total_count
    var = (total_sq_sum / total_count) - (mean**2)
    std = float(np.sqrt(max(var, 0.0)))

    # Percentili calcolati sul dataset sottocampionato
    p1 = float(np.percentile(concatenated_samples, 1))
    p2 = float(np.percentile(concatenated_samples, 2))
    p5 = float(np.percentile(concatenated_samples, 5))
    p95 = float(np.percentile(concatenated_samples, 95))
    p98 = float(np.percentile(concatenated_samples, 98))
    p99 = float(np.percentile(concatenated_samples, 99))

    stats = {
        "p2": p2,
        "p98": p98,
        "p1": p1,
        "p99": p99,
        "p5": p5,
        "p95": p95,
        "min_arcsinh": float(np.arcsinh(p1)),
        "max_arcsinh": float(np.arcsinh(p98)),
        "p5_arcsinh": float(np.arcsinh(p5)),
        "p95_arcsinh": float(np.arcsinh(p95)),
        "mean": mean,
        "std": std,
    }

    return stats


if __name__ == "__main__":
    data_dir = "./data/inputs/HI_datamix_128x128x128_stride64/val/npy_patches"
    OUTPUT_JSON = "./data/inputs/HI_datamix_128x128x128_stride64/val/dataset_stat.json"

    # subsample_stride=10 prende 1 voxel ogni 10 per i percentili
    results = compute_dataset_stats(data_dir, subsample_stride=10)

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nStatistiche salvate con successo in {OUTPUT_JSON}")