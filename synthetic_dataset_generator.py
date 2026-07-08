# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

def simulate_hi_profile(channels, central_z, w20, peak_flux):
    """Simula il profilo di una riga HI (doppia corna o gaussiana a seconda di w20)."""
    z_axis = np.arange(channels)
    # Approssimazione del profilo a doppia corna (double-horn) tipico delle galassie HI
    sigma = w20 / 3.5  # Larghezza indicativa delle corna
    dist_from_center = np.abs(z_axis - central_z)
    
    # Crea due gaussiane spostate simmetricamente dal centro per simulare la rotazione
    horn_separation = w20 / 2.0
    profile = (np.exp(-((dist_from_center - horn_separation)**2) / (2 * (sigma**0.5)**2)) + 
               np.exp(-((dist_from_center + horn_separation)**2) / (2 * (sigma**0.5)**2)))
    
    # Normalizza l'integrale del flusso
    if profile.sum() > 0:
        profile = (profile / profile.sum()) * peak_flux
    return profile

def generate_ska_hi_dataset(base_path, num_patches_x=2, num_patches_y=2, num_patches_z=2):
    """
    Genera patch sintetici 128x128x128 e un unico catalogo globale.
    """
    patch_size = 128
    
    # Crea le cartelle di output
    hr_dir = os.path.join(base_path, "hr")
    lr_dir = os.path.join(base_path, "lr")
    os.makedirs(hr_dir, exist_ok=True)
    os.makedirs(lr_dir, exist_ok=True)
    
    global_catalog = []
    
    # Simuliamo un numero di sorgenti totali nel mega-cubo
    total_possible_sources = int(num_patches_x * num_patches_y * num_patches_z * 1.5)
    
    # Generiamo a monte le proprietà fisiche delle galassie HI
    src_ids = np.arange(1000, 1000 + total_possible_sources)
    
    # Coordinate globali assolute nel mega-cubo
    max_x = num_patches_x * patch_size
    max_y = num_patches_y * patch_size
    max_z = num_patches_z * patch_size
    
    global_x = np.random.uniform(15, max_x - 15, total_possible_sources)
    global_y = np.random.uniform(15, max_y - 15, total_possible_sources)
    global_z = np.random.uniform(15, max_z - 15, total_possible_sources)
    
    # Parametri fisici astronomici coerenti
    inclinations = np.random.uniform(10.0, 85.0, total_possible_sources) # inclinazione i
    hi_sizes = np.random.exponential(scale=3.0, size=total_possible_sources) + 2.0 # hi_size
    # w20 cresce con l'inclinazione (effetto di proiezione della rotazione)
    base_rot = np.random.uniform(100.0, 300.0, total_possible_sources)
    w20_values = base_rot * np.sin(np.radians(inclinations)) # w20 in km/s (qui mappato su canali/unita)
    w20_channels = np.clip(w20_values / 10.0, 5.0, 30.0) # Convertito in scala pixel/canali per lo script
    
    line_fluxes = np.random.exponential(scale=5.0, size=total_possible_sources) + 1.0 # line_flux_integral
    central_freqs = 1420405751.0 / (1.0 + (global_z * 0.0001)) # Frequenza centrale fittizia basata su Z
    
    # Ciclo sui blocchi (Patch dello split del cubo)
    patch_counter = 0
    
    for px in range(num_patches_x):
        for py in range(num_patches_y):
            for pz in range(num_patches_z):
                
                patch_id = f"patch_{patch_counter:06d}.npy"
                
                # Limiti del patch corrente nel mega cubo
                x0, x1 = px * patch_size, (px + 1) * patch_size
                y0, y1 = py * patch_size, (py + 1) * patch_size
                z0, z1 = pz * patch_size, (pz + 1) * patch_size
                
                # Trova quali sorgenti cadono dentro questo specifico patch
                inside = (global_x >= x0) & (global_x < x1) & \
                         (global_y >= y0) & (global_y < y1) & \
                         (global_z >= z0) & (global_z < z1)
                
                indices_inside = np.where(inside)[0]
                n_sources = len(indices_inside)
                
                # Inizializza il cubo HR del patch corrente
                cube_hr = np.zeros((patch_size, patch_size, patch_size), dtype=np.float32)
                
                # Griglia spaziale interna al patch per disegnare le nubi HI estese
                X, Y = np.meshgrid(np.arange(patch_size), np.arange(patch_size), indexing='ij')
                
                # Popola il patch e aggiorna il catalogo
                for idx in indices_inside:
                    # Coordinate relative interne al patch (0-128)
                    rx = global_x[idx] - x0
                    ry = global_y[idx] - y0
                    rz = global_z[idx] - z0
                    
                    # Genera il profilo della linea lungo Z (frequenza)
                    prof_z = simulate_hi_profile(patch_size, rz, w20_channels[idx], line_fluxes[idx])
                    
                    # Distribuzione spaziale XY (estensione HI mappata come gaussiana 2D)
                    spatial_size = hi_sizes[idx]
                    g_xy = np.exp(-(((X - rx)**2 + (Y - ry)**2) / (2 * spatial_size**2)))
                    
                    # Iniezione nel cubo 3D (prodotto esterno spazio-frequenza)
                    for z_chan in range(patch_size):
                        cube_hr[z_chan] += g_xy * prof_z[z_chan]
                        
                    # Aggiungi riga al catalogo globale
                    global_catalog.append({
                        'patch_id': patch_id,
                        'n_sources_in_patch': n_sources,
                        'source_id': float(src_ids[idx]),
                        'rel_x': rx,
                        'rel_y': ry,
                        'rel_z': rz,
                        'line_flux_integral': line_fluxes[idx],
                        'hi_size': hi_sizes[idx],
                        'w20': w20_values[idx],
                        'central_freq': central_freqs[idx],
                        'i': inclinations[idx]
                    })
                
                # Se il patch e vuoto, creiamo comunque il file vuoto per consistenza del dataset
                # Generazione Cubo LR (Simulazione SKA: PSF variabile lungo i canali Z)
                cube_lr = np.zeros_like(cube_hr)
                base_psf_sigma = 3.0
                
                for z_chan in range(patch_size):
                    # Effetto SKA: la PSF varia con la frequenza (canale Z)
                    # Supponiamo frequenza decrescente lungo Z, quindi PSF cresce
                    psf_sigma = base_psf_sigma * (1.0 + (z_chan / patch_size) * 0.5)
                    
                    blurred_frame = gaussian_filter(cube_hr[z_chan], sigma=psf_sigma)
                    noise = np.random.normal(loc=0.0, scale=0.01, size=(patch_size, patch_size))
                    cube_lr[z_chan] = blurred_frame + noise
                
                # Salvataggio dei singoli patch
                np.save(os.path.join(hr_dir, patch_id), cube_hr)
                np.save(os.path.join(lr_dir, patch_id), cube_lr)
                
                print(f"Generato {patch_id} con {n_sources} sorgenti.")
                patch_counter += 1
                
    # Salvataggio del catalogo unico finale
    df_catalog = pd.DataFrame(global_catalog)
    df_catalog.to_csv(os.path.join(base_path, "global_catalog.csv"), index=False)
    print(f"\nGenerazione conclusa! Catalogo salvato in: {os.path.join(base_path, 'global_catalog.csv')}")

if __name__ == "__main__":
    # Configurazione path di output per Leonardo
    path_dataset = "/leonardo_scratch/large/userexternal/gvitanza/InverseSR/ska_hi_dataset"
    
    # 2x2x2 generera 8 patch da 128x128x128 tagliati dallo stesso cubo logico globale.
    # Puoi aumentare questi indici (es. 5, 5, 5 per avere 125 patch)
    generate_ska_hi_dataset(path_dataset, num_patches_x=5, num_patches_y=5, num_patches_z=5)