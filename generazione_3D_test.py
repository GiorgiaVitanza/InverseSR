import numpy as np
import torch
from pathlib import Path
import pandas as pd

def generate_synthetic_ska_cube_5d(
    size=128, 
    n_channels=3,
    flux_integral=1.0, 
    hi_size=30, 
    inclination=45, 
    w20=150, 
    noise_level=0.01
):
    """
    Genera un tensore 5D: (1, Canali, Z, Y, X)
    """
    # Creiamo una griglia 3D di base
    z, y, x = np.ogrid[:size, :size, :size]
    center = size // 2
    
    inc_rad = np.radians(inclination)
    cos_inc = np.cos(inc_rad)
    
    # 1. Componente Spaziale e Dinamica
    r_sq = (x - center)**2 + ((y - center) / cos_inc)**2
    spatial_disk = np.exp(-np.sqrt(r_sq) / hi_size)
    vel_gradient = (x - center) * np.sin(inc_rad) * (w20 / size)
    
    # Inizializziamo il contenitore per i canali: (Canali, Z, Y, X)
    multi_channel_cube = np.zeros((n_channels, size, size, size), dtype=np.float32)

    for c in range(n_channels):
        # Variamo leggermente i parametri per ogni canale per renderli distinti
        # Esempio: larghezza riga diversa o offset di intensità
        channel_line_width = 5.0 + (c * 2) 
        channel_flux = flux_integral * (1.0 - (c * 0.2)) # Canali via via più deboli
        
        velocity_profile = np.exp(-((z - center - vel_gradient)**2) / (2 * channel_line_width**2))
        
        cube = spatial_disk * velocity_profile
        #cube = (cube / (cube.sum() + 1e-9)) * channel_flux
        cube = (cube / (cube.max() + 1e-9)) * flux_integral
        
        if noise_level > 0:
            cube += np.random.normal(0, noise_level, cube.shape)
            
        multi_channel_cube[c] = cube

    # Aggiungiamo la dimensione del Batch -> (1, C, Z, Y, X)
    return np.expand_dims(multi_channel_cube, axis=0)

def create_test_dataset_5d(n_samples=5, size=128, output_path="./data/test_5d"):
    path = Path(output_path)
    path.mkdir(parents=True, exist_ok=True)
    catalog = []

    for i in range(n_samples):
        params = {
            "id": i,
            "flux_integral": np.random.uniform(0.5, 2.0),
            "hi_size": np.random.uniform(10, 20),
            "inclination": np.random.uniform(0, 80),
            "w20": np.random.uniform(100, 250)
        }
        
        # Generazione 5D
        cube_5d = generate_synthetic_ska_cube_5d(
            size=size,
            n_channels=3,
            **{k: v for k, v in params.items() if k != 'id'},
            noise_level=0.001
        )
        
        fname = f"ska_5d_{i:03d}.npy"
        np.save(path / fname, cube_5d)
        catalog.append(params)
        
    pd.DataFrame(catalog).to_csv(path / "ground_truth_5d.csv", index=False)
    print(f"Creati {n_samples} cubi 5D con forma: {cube_5d.shape}")

if __name__ == "__main__":
    create_test_dataset_5d()