import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pandas as pd


class RadioPatchDataset(Dataset):
    def __init__(self, data_dir, catalogue_path, in_channels):
        super().__init__()
        self.data_dir = data_dir
        self.in_channels = in_channels
        
        # Carichiamo il catalogo specifico (train o test)
        self.catalog = pd.read_csv(catalogue_path)
        self.catalog.columns = [c.lower().strip() for c in self.catalog.columns]
        
        # Fondamentale: usiamo 'patch_id' come indice perché è il nome del file
        if 'patch_id' in self.catalog.columns:
            self.catalog = self.catalog.set_index("patch_id")
        else:
            raise KeyError("Il catalogo deve contenere la colonna 'patch_id' per mappare i file .npy")

        self.feature_cols = ['hi_size', 'line_flux_integral', 'i', 'w20']

        # Statistiche per normalizzazione context (usiamo il catalogo caricato)
        self.stats = {col: (self.catalog[col].min(), self.catalog[col].max()) 
                      for col in self.feature_cols}

        # Carichiamo solo i file che sono presenti in questo specifico catalogo
        self.patch_files = self.catalog.index.tolist()

    def __len__(self):
        return len(self.patch_files)

    def __getitem__(self, idx):
        filename = self.patch_files[idx]
        
        # 1. CARICAMENTO UNICO
        path = os.path.join(self.data_dir, filename)
        data_numpy = np.load(path).astype(np.float32)

        # Rimuoviamo eventuali dimensioni batch salvate nel file (es. se era 1,128,128,128)
        if data_numpy.ndim == 4 and data_numpy.shape[0] == 1:
            data_numpy = data_numpy.squeeze(0)
        elif data_numpy.ndim == 5:
            data_numpy = data_numpy.squeeze()

        # 2. NORMALIZZAZIONE ROBUSTA (Percentile Stretching)
        p_min = data_numpy.min()
        p_max = np.percentile(data_numpy, 99.8) 
        
        x_norm = (data_numpy - p_min) / (p_max - p_min + 1e-8)
        x_0 = torch.from_numpy(np.clip(x_norm, 0, 1))

        # Assicuriamoci che abbia la forma (C, D, H, W) -> (1, 128, 128, 128)
        if x_0.ndim == 3: 
            x_0 = x_0.unsqueeze(0) 

        # Gestione Multi-Canale (se il tuo modello si aspetta 3 canali in ingresso)
        if self.in_channels == 3 and x_0.shape[0] == 1:
            x_0 = x_0.repeat(3, 1, 1, 1)

        # 3. RECUPERO CONTEXT
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

        return {
            "x_0": x_0,
            "context": context_vector
        }