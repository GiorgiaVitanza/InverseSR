import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pandas as pd
from utils.const import FITS_LIMIT, FITS_STD

class RadioPatchDataset(Dataset):
    def __init__(self, data_dir, catalogue_path, in_channels, norm_mode='global_sym'):
        """
        Args:
            data_dir: Path alle patch .npy
            catalogue_path: Path al CSV
            in_channels: Numero di canali 
            norm_mode: 'global_sym', 'local', o 'zscore'
        """
        super().__init__()
        self.data_dir = data_dir
        self.in_channels = in_channels
        self.norm_mode = norm_mode
        
        
        
        self.catalog = pd.read_csv(catalogue_path)
        self.catalog.columns = [c.lower().strip() for c in self.catalog.columns]
        
        if 'patch_id' in self.catalog.columns:
            self.catalog = self.catalog.set_index("patch_id")
        else:
            raise KeyError("Il catalogo deve contenere la colonna 'patch_id'")

        self.feature_cols = ['hi_size', 'line_flux_integral', 'i', 'w20']
        self.stats = {col: (self.catalog[col].min(), self.catalog[col].max()) 
                      for col in self.feature_cols}
        self.patch_files = self.catalog.index.tolist()

    def _normalize(self, data):
        if self.norm_mode == 'global_sym':
            # Mappa [-LIMIT, LIMIT] -> [0, 1] con zero a 0.5
            x_scaled = data / FITS_LIMIT
            return (x_scaled + 1.0) / 2.0
            
        elif self.norm_mode == 'local':
            # Stretching basato sulla singola patch
            p_min = data.min()
            p_max = np.percentile(data, 99.8) 
            return (data - p_min) / (p_max - p_min + 1e-8)
            
        elif self.norm_mode == 'zscore':
            # Standardizzazione (media 0, std 1)
            # Nota: questa non garantisce il range [0, 1]
            return (data / FITS_STD)
            
        else:
            raise ValueError(f"Modalità {self.norm_mode} non supportata.")

    def __len__(self):
        return len(self.patch_files)

    def __getitem__(self, idx):
        filename = self.patch_files[idx]
        path = os.path.join(self.data_dir, filename)
        data_numpy = np.load(path).astype(np.float32)

        if data_numpy.ndim == 4 and data_numpy.shape[0] == 1:
            data_numpy = data_numpy.squeeze(0)
        elif data_numpy.ndim == 5:
            data_numpy = data_numpy.squeeze()

        # Applicazione normalizzazione scelta
        x_norm = self._normalize(data_numpy)
        
        # Clipping finale per sicurezza (fondamentale per global_sym e local)
        x_0 = torch.from_numpy(np.clip(x_norm, 0, 1))

        if x_0.ndim == 3: 
            x_0 = x_0.unsqueeze(0) 

        if self.in_channels == 3 and x_0.shape[0] == 1:
            x_0 = x_0.repeat(3, 1, 1, 1)

        # Context (manteniamo la normalizzazione 0-1 per i parametri del catalogo)
        try:
            row = self.catalog.loc[filename]
            if isinstance(row, pd.DataFrame): 
                row = row.iloc[0] # Se ci sono più sorgenti nella stessa patch, prendi la prima
            
            params = []
            for col in self.feature_cols:
                c_min, c_max = self.stats[col]
                norm_val = (row[col] - c_min) / (c_max - c_min + 1e-8)
                params.append(np.clip(norm_val, 0, 1))
            
            context_vector = torch.tensor(params, dtype=torch.float32)
        except KeyError:
            context_vector = torch.full((len(self.feature_cols),), 0.5)

        return {"x_0": x_0, "context": context_vector}