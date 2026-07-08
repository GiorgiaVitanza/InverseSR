# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
import os

def generate_single_cube(cube_idx, size_xy=128, num_channels=128):
    """Genera una singola coppia di cubi (HR, LR) e il suo catalogo."""
    cube_hr = np.zeros((num_channels, size_xy, size_xy), dtype=np.float32)
    frequencies = np.linspace(100.0, 200.0, num_channels) 
    freq_ref = frequencies[0]
    
    # Numero casuale di sorgenti per cubo per dare variabilita al dataset
    num_sources = np.random.randint(10, 25)
    catalog_data = []
    
    for src_id in range(num_sources):
        x = np.random.uniform(10, size_xy - 10)
        y = np.random.uniform(10, size_xy - 10)
        flux_ref = np.random.exponential(scale=1.0) + 0.1
        spectral_index = np.random.normal(loc=-0.7, scale=0.15)
        is_extended = np.random.choice([0, 1], p=[0.4, 0.6])
        
        if is_extended:
            size_x = np.random.uniform(1.5, 5.0)
            size_y = np.random.uniform(1.5, 5.0)
        else:
            size_x, size_y = 0.0, 0.0
            
        catalog_data.append({
            'source_id': src_id, 'x': x, 'y': y, 'flux_ref': flux_ref,
            'spectral_index': spectral_index, 'is_extended': is_extended,
            'size_x': size_x, 'size_y': size_y
        })
        
    df_catalog = pd.DataFrame(catalog_data)
    X, Y = np.meshgrid(np.arange(size_xy), np.arange(size_xy), indexing='ij')
    
    # Popolamento HR
    for chan_idx, freq in enumerate(frequencies):
        for _, src in df_catalog.iterrows():
            flux_chan = src['flux_ref'] * ((freq / freq_ref) ** src['spectral_index'])
            if src['is_extended'] == 0:
                px, py = int(round(src['x'])), int(round(src['y']))
                cube_hr[chan_idx, px, py] += flux_chan
            else:
                g = flux_chan * np.exp(-(((X - src['x'])**2 / (2 * src['size_x']**2)) + 
                                         ((Y - src['y'])**2 / (2 * src['size_y']**2))))
                cube_hr[chan_idx] += g

    # Generazione LR (Degradazione SKA-like)
    cube_lr = np.zeros_like(cube_hr)
    base_psf_sigma = 3.5  
    
    for chan_idx, freq in enumerate(frequencies):
        psf_sigma = base_psf_sigma * (frequencies[-1] / freq)
        blurred_channel = gaussian_filter(cube_hr[chan_idx], sigma=psf_sigma)
        noise_sigma = 0.02
        noise = np.random.normal(loc=0.0, scale=noise_sigma, size=(size_xy, size_xy))
        cube_lr[chan_idx] = blurred_channel + noise

    return cube_hr, cube_lr, df_catalog

def generate_dataset(base_path, split_name, num_cubes):
    """Genera una quantita definita di cubi per un determinato split (train/val/test)."""
    print(f"\n--- Generazione split: {split_name.upper()} ({num_cubes} cubi) ---")
    
    # Crea le sottocartelle dedicate
    hr_dir = os.path.join(base_path, split_name, "hr")
    lr_dir = os.path.join(base_path, split_name, "lr")
    cat_dir = os.path.join(base_path, split_name, "catalogs")
    
    for folder in [hr_dir, lr_dir, cat_dir]:
        os.makedirs(folder, exist_ok=True)
        
    for i in range(num_cubes):
        cube_id = i + 1
        hr, lr, cat = generate_single_cube(cube_idx=cube_id)
        
        # Salvataggio con nomi coerenti indicizzati
        np.save(os.path.join(hr_dir, f"cube_{cube_id:03d}_hr.npy"), hr)
        np.save(os.path.join(lr_dir, f"cube_{cube_id:03d}_lr.npy"), lr)
        cat.to_csv(os.path.join(cat_dir, f"catalog_{cube_id:03d}.csv"), index=False)
        
        if cube_id % 10 == 0 or cube_id == num_cubes:
            print(f"Avanzamento {split_name}: {cube_id}/{num_cubes} cubi completati.")

if __name__ == "__main__":
    # Configura qui la cartella di output sul tuo scratch di Leonardo
    base_dataset_path = "/leonardo_scratch/large/userexternal/gvitanza/InverseSR/ska_dataset"
    
    # Definisci quanti cubi vuoi per ogni partizione
    # Per iniziare puoi fare un dataset piccolo, poi aumenti i numeri per il cluster
    num_train = 80
    num_val = 15
    num_test = 5
    
    generate_dataset(base_dataset_path, "train", num_train)
    generate_dataset(base_dataset_path, "val", num_val)
    generate_dataset(base_dataset_path, "test", num_test)
    
    print(f"\nDataset completo generato con successo in: {base_dataset_path}")