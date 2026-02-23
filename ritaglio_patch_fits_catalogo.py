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


def process_radio_multiformat(fits_path, catalog_path, output_dir, 
                             patch_size=(128, 128, 128), stride=128, 
                             output_format='npy'):
    """
    Args:
        output_format (str): 'fits', 'npy', o 'both'
    """
    print(f"Apertura FITS: {fits_path}")
    hdul = fits.open(fits_path, memmap=True, mode='readonly')
    header = hdul[0].header
    wcs = WCS(header)
    if wcs.naxis == 4: wcs = wcs.dropaxis(3)

    raw_data_ref = hdul[0].data
    Z, Y, X = (raw_data_ref.shape[1:] if len(raw_data_ref.shape)==4 else raw_data_ref.shape)

    print(f"Caricamento catalogo: {catalog_path}")
    df_cat = pd.read_table(catalog_path, sep='\s+')

    # Conversione RA/Dec/Freq -> Pixel
    sky_coords = df_cat[['ra', 'dec', 'central_freq']].values
    pixels = wcs.all_world2pix(sky_coords, 0)
    df_cat['x_pix'], df_cat['y_pix'], df_cat['z_pix'] = pixels[:, 0], pixels[:, 1], pixels[:, 2]

    # Setup cartelle di output
    paths = {}
    if output_format in ['fits', 'both']:
        paths['fits'] = os.path.join(output_dir, "fits_patches")
        os.makedirs(paths['fits'], exist_ok=True)
    if output_format in ['npy', 'both']:
        paths['npy'] = os.path.join(output_dir, "npy_patches")
        os.makedirs(paths['npy'], exist_ok=True)

    all_patches_list = []
    patch_count = 0

    for z in tqdm(range(0, Z - patch_size[0] + 1, stride), desc="Z-axis"):
        cat_z = df_cat[(df_cat['z_pix'] >= z) & (df_cat['z_pix'] < z + patch_size[0])]
        if cat_z.empty: continue

        for y in range(0, Y - patch_size[1] + 1, stride):
            for x in range(0, X - patch_size[2] + 1, stride):
                sources = cat_z[(cat_z['y_pix'] >= y) & (cat_z['y_pix'] < y + patch_size[1]) &
                                (cat_z['x_pix'] >= x) & (cat_z['x_pix'] < x + patch_size[2])].copy()
                
                if not sources.empty:
                    base_name = f"patch_{patch_count:06d}"
                    
                    # Estrazione dati
                    d_slice = raw_data_ref[0, z:z+patch_size[0], y:y+patch_size[1], x:x+patch_size[2]] if len(raw_data_ref.shape)==4 else raw_data_ref[z:z+patch_size[0], y:y+patch_size[1], x:x+patch_size[2]]
                    p_data = np.nan_to_num(np.array(d_slice, dtype=np.float32), nan=0.0)
                    
                    # 1. Salvataggio FITS
                    if 'fits' in paths:
                        p_name_fits = base_name + ".fits"
                        phdu = fits.PrimaryHDU(data=p_data, header=header)
                        for i, coord in enumerate([x, y, z]):
                            key = f'CRPIX{i+1}'
                            if key in phdu.header: phdu.header[key] -= coord
                        phdu.writeto(os.path.join(paths['fits'], p_name_fits), overwrite=True)
                    
                    # 2. Salvataggio NPY (Formato ML: 1, D, H, W)
                    if 'npy' in paths:
                        p_name_npy = base_name + ".npy"
                        # Aggiungiamo la dimensione del canale per compatibilità DL
                        np.save(os.path.join(paths['npy'], p_name_npy), p_data[np.newaxis, ...])
                    
                    # Update catalogo (usiamo il nome base o quello scelto)
                    sources['patch_id'] = base_name + (".fits" if output_format=='fits' else ".npy")
                    sources['rel_x'], sources['rel_y'], sources['rel_z'] = sources['x_pix']-x, sources['y_pix']-y, sources['z_pix']-z
                    all_patches_list.append(sources)
                    patch_count += 1

    hdul.close()
    master_path = os.path.join(output_dir, "master_patch_catalog.csv")
    pd.concat(all_patches_list).to_csv(master_path, index=False)
    print(f"\nProcesso completato. {patch_count} patch salvati in formato {output_format}.")
    return output_dir, master_path

# --- ESECUZIONE ---
if __name__ == "__main__":
    OUT_DIR = "./data/processed_dataset"
    
    # Scegli qui: 'fits', 'npy', o 'both'
    FORMATO = 'npy' 

    # process_radio_multiformat(
    #     fits_path="./data/inputs/sky_dev_v2.fits",
    #     catalog_path="./data/inputs/sky_dev_truthcat_v2.txt",
    #     output_dir=OUT_DIR,
    #     patch_size=(128, 128, 128),
    #     stride=128,
    #     output_format=FORMATO
    # )

    visualize_patch_multi_view(f"{OUT_DIR}/npy_patches", "patch_000005.npy", os.path.join(OUT_DIR, "master_patch_catalog.csv"))