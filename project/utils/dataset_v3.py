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
        self.catalog = pd.read_csv(catalogue_path, sep=",")
        self.catalog.columns = [c.lower().strip() for c in self.catalog.columns]

        if 'id' in self.catalog.columns:
            self.catalog['id'] = self.catalog['id'].astype(str)
        else:
            raise KeyError(f"Colonna 'id' non trovata.")
        
        self.catalog = self.catalog.set_index("id")
        self.feature_cols = ['hi_size', 'line_flux_integral', 'i', 'w20']

        # --- STATISTICHE PER NORMALIZZAZIONE 0-1 ---
        # 1. Statistiche Dati Volumetrici (Valori basati sul tuo dataset radio)
        # Nota: Sostituisci questi valori con il min/max reali del tuo set di training
        self.data_min = -1.47367257e-03  # Esempio: rumore di fondo minimo
        self.data_max = 1.52088422e-03     # Esempio: picco di intensità massima
        
        # 2. Statistiche Catalogo (Calcolate una volta sola)
        self.stats = {col: (self.catalog[col].min(), self.catalog[col].max()) 
                      for col in self.feature_cols}

        self.patch_files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]

    def normalize_01(self, x, v_min, v_max):
        """Helper per normalizzare tra 0 e 1 con clipping"""
        x = (x - v_min) / (v_max - v_min + 1e-8)
        return torch.clamp(x, 0, 1)

    def __len__(self):
        return len(self.patch_files)

    def __getitem__(self, idx):
        filename = self.patch_files[idx]
        galaxy_id = filename.split('_')[0]
        
        # --- CARICAMENTO E NORMALIZZAZIONE CUBO ---
        data_numpy = np.load(os.path.join(self.data_dir, filename))
        x_0 = torch.from_numpy(data_numpy.astype(np.float32))
        
        if x_0.ndim == 5: x_0 = x_0.squeeze(0)
        if x_0.ndim == 3: x_0 = x_0.unsqueeze(0)

        # Normalizzazione 0-1 del cubo
        x_0 = self.normalize_01(x_0, self.data_min, self.data_max)

        # Gestione Canali (Replica se necessario)
        if self.in_channels == 3 and x_0.shape[0] == 1:
            x_0 = x_0.repeat(3, 1, 1, 1)

        # --- RECUPERO E NORMALIZZAZIONE CONTEXT ---
        try:
            row = self.catalog.loc[galaxy_id]
            params = []
            for col in self.feature_cols:
                c_min, c_max = self.stats[col]
                # Normalizzazione 0-1 della singola feature
                norm_val = (row[col] - c_min) / (c_max - c_min + 1e-8)
                params.append(np.clip(norm_val, 0, 1)) # Clipping di sicurezza
            
            context_vector = torch.tensor(params, dtype=torch.float32)
        except KeyError:
            # Se la galassia manca, restituiamo un vettore neutro (es. 0.5)
            context_vector = torch.full((len(self.feature_cols),), 0.5)

        return {
            "x_0": x_0,              # Range [0, 1], Shape (C, D, H, W)
            "context": context_vector # Range [0, 1], Shape (4,)
        }
