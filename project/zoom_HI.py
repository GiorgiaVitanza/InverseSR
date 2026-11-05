import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits, ascii
from astropy.io.votable import parse_single_table
from astropy.wcs import WCS
from astropy.table import Table
from astropy.visualization import ImageNormalize, AsinhStretch, PercentileInterval
import torch
import warnings
import gc
from astropy.utils.exceptions import AstropyUserWarning
import pyvista as pv

warnings.simplefilter('ignore', category=AstropyUserWarning)
pv.OFF_SCREEN = True

def load_catalog_universal(catalog_path):
    """
    Carica un catalogo in qualsiasi formato (.xml/.vot, .txt, .csv, .fits).
    Ritorna un oggetto astropy.table.Table con le colonne normalizzate.
    """
    print(f"📄 Caricamento catalogo sorgenti: {catalog_path}")
    
    # 1. Gestione specifica per formati XML / VOTable
    if catalog_path.lower().endswith(('.xml', '.vot', '.votable')):
        try:
            votable = parse_single_table(catalog_path)
            cat = votable.to_table()
        except Exception as e:
            print(f"⚠️ parsing VOTable fallito, tentiamo fallback generico: {e}")
            cat = Table.read(catalog_path, format='votable')
    else:
        # 2. Gestione generica per TXT, CSV, FITS
        try:
            cat = ascii.read(catalog_path, guess=True)
        except Exception:
            cat = Table.read(catalog_path)

    # Pulizia universale dei nomi colonna: rimuove '#', spazi e forza il minuscolo
    for col in list(cat.colnames):
        clean_col = col.replace('#', '').strip().lower()
        if clean_col != col:
            cat.rename_column(col, clean_col)

    print(f"📋 Nomi colonne rilevati e puliti ({len(cat.colnames)} colonne): {cat.colnames[:10]}...")
    return cat

def get_source_pixel_coords(catalog_path, wcs, data_3d, source_id=None, name = "EMU"):
    """
    Estrae le coordinate (RA, Dec) dal catalogo (XML/TXT/CSV/FITS), 
    converte in Pixel X, Y e individua il canale Z di picco reale.
    """
    cat = load_catalog_universal(catalog_path)

    # Identificazione della colonna ID
    id_candidates = ['id', 'source_id', 'src_id', 'name', 'source_name']
    id_col = next((c for c in cat.colnames if c in id_candidates), cat.colnames[0])

    # Selezione della riga della sorgente
    if source_id is not None:
        mask = [str(val).strip() == str(source_id).strip() for val in cat[id_col]]
        row = cat[mask][0] if any(mask) else cat[0]
        if not any(mask):
            print(f"⚠️ ID '{source_id}' non trovato. Selezione prima sorgente (ID: {cat[0][id_col]}).")
        else:
            print(f"🎯 Sorgente selezionata ID: {row[id_col]}")
    else:
        row = cat[0]
        print(f"🎯 Selezione automatica prima sorgente (ID: {row[id_col]})")

    # Identificazione flessibile delle colonne RA e DEC
    ra_candidates = ['ra', 'ra_deg', 'ra_degree', 'coord_ra', 'x_world', 'pos_ra']
    dec_candidates = ['dec', 'dec_deg', 'dec_degree', 'coord_dec', 'y_world', 'pos_dec']

    ra_col = next((c for c in cat.colnames if c in ra_candidates), None)
    dec_col = next((c for c in cat.colnames if c in dec_candidates), None)

    if not ra_col or not dec_col:
        raise KeyError(f"❌ Impossibile identificare le colonne RA/DEC nel catalogo. Colonne: {cat.colnames}")

    ra_val, dec_val = float(row[ra_col]), float(row[dec_col])

    # 1. Conversione RA/Dec -> Pixel X, Y
    x_pix, y_pix = wcs.celestial.world_to_pixel_values(ra_val, dec_val)
    x_pix, y_pix = int(round(float(x_pix))), int(round(float(y_pix)))

    # Protezione limiti della griglia del cubo
    n_chan, ny, nx = data_3d.shape
    x_pix, y_pix = max(0, min(nx - 1, x_pix)), max(0, min(ny - 1, y_pix))

    # 2. Ricerca del canale Z (Velocità/Frequenza) al picco di flusso reale
    spectrum_los = np.nan_to_num(data_3d[:, y_pix, x_pix], nan=0.0)
    z_chan = int(np.argmax(spectrum_los))
    max_flux = spectrum_los[z_chan]

    print(f"📍 Sky Coords: RA={ra_val:.5f}°, DEC={dec_val:.5f}°")
    print(f"🎯 Pixel Spaziali: X={x_pix}, Y={y_pix}")
    print(f"📻 Canale Z individuato dal picco: {z_chan} (Flusso picco: {max_flux:.5e})")

    return x_pix, y_pix, z_chan

def visualize_source_from_catalog(fits_filepath, catalog_filepath, source_id=None, box_size_xy=50, box_size_z=30, name = "EMU"):
    hdul = fits.open(fits_filepath, memmap=True)
    raw_data = hdul[0].data
    header = hdul[0].header
    
    shape = [s for s in raw_data.shape if s > 1]
    data_3d = raw_data.reshape(shape)
    n_chan, ny, nx = data_3d.shape
    
    wcs = WCS(header)

    x_src, y_src, z_src = get_source_pixel_coords(catalog_filepath, wcs, data_3d, source_id=source_id, name = name)

    x_min, x_max = max(0, x_src - box_size_xy), min(nx, x_src + box_size_xy)
    y_min, y_max = max(0, y_src - box_size_xy), min(ny, y_src + box_size_xy)
    z_min, z_max = max(0, z_src - box_size_z), min(n_chan, z_src + box_size_z)

    print(f"✂️ Crop attorno alla sorgente:")
    print(f"   X (Pixel) : [{x_min}:{x_max}]")
    print(f"   Y (Pixel) : [{y_min}:{y_max}]")
    print(f"   Z (Canali): [{z_min}:{z_max}]")

    sub_cube = np.nan_to_num(data_3d[z_min:z_max, y_min:y_max, x_min:x_max], nan=0.0).astype(np.float32)

    # MAPPA 2D MOMENTO 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sub_tensor = torch.from_numpy(sub_cube).to(device)
    
    sigma_loc = float(np.nanstd(sub_cube))
    sub_tensor[sub_tensor < 3.0 * sigma_loc] = 0.0
    
    mom0_crop = torch.sum(sub_tensor, dim=0).cpu().numpy()
    del sub_tensor
    torch.cuda.empty_cache()

    wcs_crop = wcs.celestial[y_min:y_max, x_min:x_max]

    norm = ImageNormalize(mom0_crop, interval=PercentileInterval(99.5), stretch=AsinhStretch(a=0.1))

    fig = plt.figure(figsize=(12, 6))
    ax_map = fig.add_subplot(121, projection=wcs_crop)
    ax_spec = fig.add_subplot(122)

    im = ax_map.imshow(mom0_crop, origin='lower', cmap='magma', norm=norm)
    ax_map.set_title(f"Sorgente (Mappa Mom0 2D)")
    ax_map.set_xlabel("RA (J2000)")
    ax_map.set_ylabel("Dec (J2000)")
    fig.colorbar(im, ax=ax_map, label='Intensità Integrata')

    x_rel, y_rel = x_src - x_min, y_src - y_min
    ax_map.plot(x_rel, y_rel, 'cyan', marker='x', markersize=12, markeredgewidth=2)

    spectrum_source = sub_cube[:, y_rel, x_rel]
    channels_crop = np.arange(z_min, z_max)
    ax_spec.plot(channels_crop, spectrum_source, color='navy', lw=1.5)
    ax_spec.axvline(z_src, color='red', linestyle='--', label=f'Peak Z={z_src}')
    ax_spec.set_title(f"Spettro HI Sorgente (X:{x_src}, Y:{y_src})")
    ax_spec.set_xlabel("Canale")
    ax_spec.set_ylabel("Flusso")
    ax_spec.grid(True, linestyle='--', alpha=0.6)
    ax_spec.legend()

    plt.tight_layout()
    output_2d = f"sorgente_{source_id if source_id else 'crop'}_2D.png"
    plt.savefig(output_2d, dpi=300)
    plt.close()
    print(f"📸 Salvata mappa 2D sorgente: {output_2d}")

    # RENDERING 3D PYVISTA
    print("🧊 Generazione volume 3D con pulizia del rumore in PyVista...")
    threshold_loc = 3.0 * sigma_loc
    sub_cube_clean = np.where(sub_cube > threshold_loc, sub_cube, 0.0)
    sub_cube_log = np.log1p(np.maximum(sub_cube_clean, 0))

    grid_crop = pv.ImageData()
    grid_crop.dimensions = sub_cube_log.shape[::-1]
    grid_crop.spacing = (1.0, 1.0, 1.0)
    grid_crop.point_data["HI_source_log"] = sub_cube_log.flatten(order="C")

    opacity_transfer_function = [0.0, 0.0, 0.05, 0.3, 0.8, 1.0]

    plotter = pv.Plotter(off_screen=True)
    plotter.add_volume(
        grid_crop, 
        scalars="HI_source_log", 
        cmap="plasma", 
        opacity=opacity_transfer_function
    )
    plotter.add_axes()
    plotter.show_grid()
    
    output_3d = f"sorgente_{source_id if source_id else 'crop'}_3D.png"
    plotter.screenshot(output_3d)
    plotter.close()
    
    hdul.close()
    print(f"📸 Salvato rendering 3D sorgente: {output_3d}")

if __name__ == "__main__":
    mio_file_fits = "/leonardo_scratch/large/userexternal/gvitanza/InverseSR/data/inputs/image.i.EMU_0900-28.SB88648.cont.taylor.0.restored.conv.fits"
    mio_catalogo = "/leonardo_scratch/large/userexternal/gvitanza/InverseSR/data/inputs/selavy-image.i.EMU_0900-28.SB88648.cont.taylor.0.restored.conv.components.xml"  
    name = "EMU"
    visualize_source_from_catalog(
        fits_filepath=mio_file_fits, 
        catalog_filepath=mio_catalogo,
        source_id=None,
        box_size_xy=50,
        box_size_z=30,
        name=name,
    )