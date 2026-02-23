import torch
import numpy as np
from astropy.io import fits
from monai.transforms import (
    MapTransform,
    EnsureChannelFirstd,
    Compose,
    ScaleIntensityRangePercentilesd, # Molto meglio per Astro
    CenterSpatialCropd,              # Meglio di SpatialCropd fisso
    SpatialPadd,
    ToTensord,
    RandFlipd,
    RandRotate90d
)

class LoadFitsd(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img_path = d[key]
            
            try:
                with fits.open(img_path) as hdul:
                    raw_data = None
                    found_ext = -1

                    # 1. CERCA I DATI
                    for i, hdu in enumerate(hdul):
                        if hdu.data is not None and hdu.data.size > 0 and len(hdu.data.shape) >= 2:
                            # CRUCIALE: .copy() sposta i dati dalla mmap alla RAM reale
                            raw_data = hdu.data.copy()
                            found_ext = i
                            break 
                
                    if raw_data is None:
                        raise ValueError(f"Nessun dato valido trovato in {img_path}")

                    # 2. GESTIONE ENDIANNESS (FITS è Big-Endian, PyTorch vuole Little-Endian)
                    # Senza questo, PyTorch darà errore: "Stride is negative" o performance lente
                    if raw_data.dtype.byteorder == '>' or (raw_data.dtype.byteorder == '=' and np.little_endian == False):
                        raw_data = raw_data.byteswap().newbyteorder()

                    # 3. CLEANING (NaN / Inf)
                    # I FITS hanno spesso NaN ai bordi o Inf per divisioni per zero
                    raw_data = np.nan_to_num(raw_data, nan=0.0, posinf=raw_data.max(), neginf=raw_data.min())

                    # 4. CASTING
                    d[key] = raw_data.astype(np.float32)
                    
                    # Debug print (utile all'inizio, poi commentalo)
                    # print(f"Loaded {img_path}: {d[key].shape}, Ext: {found_ext}")

            except Exception as e:
                print(f"Errore caricando {img_path}: {e}")
                raise e

        return d

# 2. Pipeline Pipeline Adattata
def get_preprocessing(device: torch.device) -> Compose:
    # Definisci la dimensione target del cubo (es. 128x128x128 o 64x64x64)
    # Deve essere potenza di 2 per le UNet standard
    CUBE_SIZE = (64, 64, 64) 

    return Compose(
        [
            LoadFitsd(keys=["image"]),
            
            # Aggiunge canale -> (1, Depth, Height, Width)
            EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
            
            # --- NORMALIZZAZIONE ROBUSTA ---
            # Invece di MinMax puro, tagliamo il rumore estremo (es. RFI) e i pixel hot
            # Taglia tra il 1° e il 99.9° percentile e scala a [0, 1]
            ScaleIntensityRangePercentilesd(
                keys=["image"], 
                lower=1, 
                upper=99.9, 
                b_min=0.0, 
                b_max=1.0, 
                clip=True
            ),
            
            # --- GEOMETRIA ---
            # CenterSpatialCrop è più sicuro per oggetti astronomici
            CenterSpatialCropd(
                keys=["image"],
                roi_size=CUBE_SIZE, 
            ),
            
            # Se il cubo è più piccolo di CUBE_SIZE, aggiunge zeri attorno
            SpatialPadd(
                keys=["image"],
                spatial_size=CUBE_SIZE,
                method="symmetric" # o "constant" (zeri)
            ),
            
            # --- DATA AUGMENTATION (Opzionale ma raccomandata) ---
            # Utile perché nello spazio non c'è "su" o "giù"
            # RandRotate90d(keys=["image"], prob=0.5, spatial_axes=(1, 2)), # Ruota su piano cielo
            # RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
            
            ToTensord(keys=["image"], device=device),
        ]
    )