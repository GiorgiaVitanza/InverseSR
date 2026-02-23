import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pandas as pd

class RadioPatchDataset(Dataset):
    def __init__(self, data_dir, catalogue_path):
        super().__init__()
        self.data_dir = data_dir

        # 1. CARICAMENTO CATALOGO
        self.catalog = pd.read_csv(catalogue_path, sep="\s+")
        self.catalog['id'] = self.catalog['id'].astype(str)
        self.catalog = self.catalog.set_index("id")

        # Statistiche per NORMALIZZAZIONE
        self.freq_min = self.catalog['central_freq'].min()
        self.freq_max = self.catalog['central_freq'].max()
        self.flux_min = self.catalog['line_flux_integral'].min()
        self.flux_max = self.catalog['line_flux_integral'].max()

        self.patch_files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
        print(f"Dataset inizializzato: {len(self.patch_files)} file trovati.")

    def __len__(self):
        return len(self.patch_files)

    def __getitem__(self, idx):
        filename = self.patch_files[idx]
        filepath = os.path.join(self.data_dir, filename)
        galaxy_id = filename.split('_')[0]

        # -----------------------------------------------------------
        # A. CARICAMENTO GROUND TRUTH (x_0)
        # -----------------------------------------------------------
        try:
            data_numpy = np.load(filepath)
            data_tensor = torch.from_numpy(data_numpy.astype(np.float32))

            # Gestione dimensioni iniziali (es. se è 5D [1, C, D, H, W])
            if data_tensor.ndim == 5:
                data_tensor = data_tensor.squeeze(0) # Diventa (C, D, H, W)
            
            # normalizzazione
            data_tensor = (data_tensor - data_tensor.min()) / (data_tensor.max() - data_tensor.min() + 1e-8)
            # verifica range [0, 1]
            if not (0 <= data_tensor.min() and data_tensor.max() <= 1):
                raise ValueError(f"Data normalization failed for file {filename}: min {data_tensor.min()}, max {data_tensor.max()}")
            if data_tensor.shape[0] >= 1:
                # Prendiamo il primo canale: (1, D, H, W)
                data_tensor = data_tensor[0:1] 
            
            # Controllo finale: deve essere 1 canale
            if data_tensor.shape[0] != 1:
                raise ValueError(f"Atteso 1 canale, ottenuto {data_tensor.shape[0]}")

        except Exception as e:
            print(f"Errore file {filename}: {e}")
            return self.__getitem__((idx + 1) % len(self))

        # -----------------------------------------------------------
        # B. RECUPERO E NORMALIZZAZIONE PARAMETRI FISICI
        # -----------------------------------------------------------
        try:
            row = self.catalog.loc[galaxy_id]
            raw_freq = row['central_freq']
            raw_flux = row['line_flux_integral']
        except KeyError:
            raw_freq = self.freq_min
            raw_flux = self.flux_min

        norm_freq = (raw_freq - self.freq_min) / (self.freq_max - self.freq_min + 1e-8)
        norm_flux = (raw_flux - self.flux_min) / (self.flux_max - self.flux_min + 1e-8)

        # Broadcasting spaziale (Creiamo le mappe piene)
        spatial_dims = data_tensor.shape[1:] # (D, H, W)
        freq_channel = torch.full((1, *spatial_dims), norm_freq, dtype=torch.float32)
        flux_channel = torch.full((1, *spatial_dims), norm_flux, dtype=torch.float32)

        # Physics maps rimane a 2 canali
        phys_maps = torch.cat([freq_channel, flux_channel], dim=0)

        return {
            "x_0": data_tensor,   # ORA È (3, D, H, W)
            "physics": phys_maps  # (2, D, H, W)
        }
