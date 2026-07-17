import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits

from sklearn.model_selection import train_test_split

def split_original_catalog(catalog_path, train_ratio=0.8, seed=42):
    df_cat = pd.read_table(catalog_path, sep='\s+')
    
    train_df, test_df = train_test_split(
        df_cat, 
        train_size=train_ratio, 
        random_state=seed,
        shuffle=True
    )
    
    print(f"Split completato: {len(train_df)} train, {len(test_df)} test")
    return train_df, test_df

def visualize_patch_multi_view(patch_dir, patch_id, master_catalog_path):
    df = pd.read_csv(master_catalog_path)
    sources = df[df['patch_id'] == patch_id].reset_index()
    n_sources = len(sources)

    if n_sources == 0: return

    # Caricamento dati
    data = np.squeeze(np.load(os.path.join(patch_dir, patch_id))) # o fits

    # Creiamo una riga di subplot, uno per sorgente
    fig, axes = plt.subplots(1, n_sources, figsize=(5 * n_sources, 5), squeeze=False)

    for i, (_, row) in enumerate(sources.iterrows()):
        ax = axes[0, i]
        z_idx = int(round(row['rel_z']))
        z_idx = np.clip(z_idx, 0, data.shape[0]-1)
        
        slice_2d = data[z_idx, :, :]
        im = ax.imshow(slice_2d, origin='lower', cmap='hot', vmax=np.percentile(slice_2d, 99))
        
        # Evidenziamo la sorgente "corrente" del subplot
        ax.scatter(row['rel_x'], row['rel_y'], s=150, edgecolors='cyan', facecolors='none', lw=3)
        ax.set_title(f"Source ID: {int(row['id'])}\nSlice Z: {z_idx}")

    plt.tight_layout()
    plt.show()

def process_radio_multiformat(fits_path, df_cat, output_dir, subset_name="train", 
                             patch_size=(128, 128, 128), stride=128, 
                             output_format='npy'):
    
    print(f"Apertura FITS: {fits_path}")
    hdul = fits.open(fits_path, memmap=True, mode='readonly')
    header_originale = hdul[0].header
    wcs = WCS(header_originale)
    if wcs.naxis == 4: wcs = wcs.dropaxis(3)

    raw_data_ref = hdul[0].data
    # Gestione dimensioni (Stokes, Freq, Dec, RA) -> (Z, Y, X)
    shape = raw_data_ref.shape
    Z, Y, X = (shape[1:] if len(shape)==4 else shape)

    print(f"Caricamento catalogo: {len(df_cat)} sorgenti")
    df_cat = df_cat.copy()

    
    # (id, ra, dec, hi_size, line_flux_integral, central_freq, pa, i, w20)
    col_ra = 'ra'
    col_dec = 'dec'
    col_freq = 'central_freq'
    col_flux = 'line_flux_integral'

    # Conversione RA/Dec/Freq -> Pixel
    sky_coords = df_cat[[col_ra, col_dec, col_freq]].values
    pixels = wcs.all_world2pix(sky_coords, 0)
    df_cat['x_pix'], df_cat['y_pix'], df_cat['z_pix'] = pixels[:, 0], pixels[:, 1], pixels[:, 2]


    # Creiamo una sottocartella specifica per il subset (es. ./data/train/...)
    subset_dir = os.path.join(output_dir, subset_name)
    paths = {fmt: os.path.join(subset_dir, f"{fmt}_patches") 
             for fmt in (['fits', 'npy'] if output_format == 'both' else [output_format])}
    for p in paths.values(): os.makedirs(p, exist_ok=True)

    

    master_records = []
    patch_count = 0

    for z in tqdm(range(0, Z - patch_size[0] + 1, stride), desc="Z-axis"):
        cat_z = df_cat[(df_cat['z_pix'] >= z) & (df_cat['z_pix'] < z + patch_size[0])]
        if cat_z.empty: continue

        for y in range(0, Y - patch_size[1] + 1, stride):
            for x in range(0, X - patch_size[2] + 1, stride):
                
                # Filtro sorgenti nel patch
                sources = cat_z[(cat_z['y_pix'] >= y) & (cat_z['y_pix'] < y + patch_size[1]) &
                                (cat_z['x_pix'] >= x) & (cat_z['x_pix'] < x + patch_size[2])].copy()
                

                if not sources.empty:
                    base_name = f"patch_{patch_count:06d}"
                    
                    # 1. Estrazione dati (gestione 3D o 4D)
                    if len(raw_data_ref.shape) == 4:
                        d_slice = raw_data_ref[0, z:z+patch_size[0], y:y+patch_size[1], x:x+patch_size[2]]
                    else:
                        d_slice = raw_data_ref[z:z+patch_size[0], y:y+patch_size[1], x:x+patch_size[2]]
                    
                    p_data = np.nan_to_num(np.array(d_slice, dtype=np.float32), nan=0.0)
                    
                    # 2. Aggiornamento Header FITS con Astropy
                    patch_header = header_originale.copy()
                    patch_header['NAXIS1'] = patch_size[2]
                    patch_header['NAXIS2'] = patch_size[1]
                    patch_header['NAXIS3'] = patch_size[0]
                    
                    # Spostiamo il riferimento WCS per il nuovo ritaglio
                    patch_header['CRPIX1'] -= x
                    patch_header['CRPIX2'] -= y
                    patch_header['CRPIX3'] -= z

                    # 3. Salvataggio (FITS / NPY)
                    if 'fits' in paths:
                        fits.writeto(os.path.join(paths['fits'], f"{base_name}.fits"), 
                                     p_data, patch_header, overwrite=True)
                    if 'npy' in paths:
                        # Salvataggio per PyTorch (C, D, H, W)
                        np.save(os.path.join(paths['npy'], f"{base_name}.npy"), p_data[np.newaxis, ...])
                    
                    # 4. Selezione della sorgente principale nel patch
                    # Usiamo 'line_flux_integral' per decidere qual è la sorgente dominante
                    main_source = sources.loc[sources[col_flux].idxmax()].copy()
                    
                    # Creiamo il record unico per questo patch
                    record = {
                        'patch_id': f"{base_name}.{output_format if output_format != 'both' else 'npy'}",
                        'n_sources_in_patch': len(sources),
                        'source_id': main_source['id'],
                        'rel_x': main_source['x_pix'] - x,
                        'rel_y': main_source['y_pix'] - y,
                        'rel_z': main_source['z_pix'] - z,
                        # Valori di condizionamento numerico
                        'line_flux_integral': main_source[col_flux],
                        'hi_size': main_source['hi_size'],
                        'w20': main_source['w20'],
                        'central_freq': main_source[col_freq],
                        'i': main_source['i'],
                    }
                    master_records.append(record)
                    patch_count += 1
                
    
    hdul.close()
    # Salvataggio catalogo specifico per questo subset
    master_df = pd.DataFrame(master_records)
    master_df.to_csv(os.path.join(subset_dir, f"{subset_name}_catalog.csv"), index=False)
    return output_dir, master_df

if __name__ == "__main__":
    FITS_PATH = "./data/inputs/cont_dev.fits"
    CATALOG_PATH = "./data/inputs/sky_dev_truthcat_v2.txt"
    BASE_OUT_DIR = "./data/inputs/16x128x128_stride128_cont_dev" # Cambia questo percorso se vuoi un output diverso
    
    # 1. Split delle sorgenti a monte
    train_cat, test_cat = split_original_catalog(CATALOG_PATH, train_ratio=0.8)

    # 2. Processamento Train Set
    print("\n--- Processing TRAIN SET ---")
    process_radio_multiformat(
        fits_path=FITS_PATH,
        df_cat=train_cat,
        output_dir=BASE_OUT_DIR,
        subset_name="train",
        patch_size=(16, 128, 128),
        stride=128, # Stride più piccolo per fare data augmentation nel train
        output_format='npy'
    )

    # 3. Processamento Test Set
    print("\n--- Processing TEST SET ---")
    process_radio_multiformat(
        fits_path=FITS_PATH,
        df_cat=test_cat,
        output_dir=BASE_OUT_DIR,
        subset_name="test",
        patch_size=(16, 128, 128),
        stride=128, # Stride pieno per il test (meno ridondanza)
        output_format='npy'
    )
    #visualize_patch_multi_view(f"{OUT_DIR}/npy_patches", "patch_000005.npy", os.path.join(OUT_DIR, "master_patch_catalog.csv"))
