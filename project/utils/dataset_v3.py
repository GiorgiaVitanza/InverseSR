import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pandas as pd

class RadioPatchDataset(Dataset):
    def __init__(self, data_dir, catalogue_path):
        super().__init__()
        self.data_dir = data_dir
        self.catalog = pd.read_csv(catalogue_path, sep="\s+")
        self.catalog['id'] = self.catalog['id'].astype(str)
        self.catalog = self.catalog.set_index("id")

        # Selezioniamo le 4 colonne che avevamo concordato
        self.feature_cols = ['hi_size', 'line_flux_integral', 'i', 'w20']
        
        # Calcoliamo min e max globali per una normalizzazione coerente
        self.stats = {col: (self.catalog[col].min(), self.catalog[col].max()) 
                      for col in self.feature_cols}

        self.patch_files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
    
    def __len__(self):
        return len(self.patch_files)
    
    def __getitem__(self, idx):
        filename = self.patch_files[idx]
        galaxy_id = filename.split('_')[0]
        
        # 1. Caricamento Cubo (x_0)
        data_numpy = np.load(os.path.join(self.data_dir, filename))
        x_0 = torch.from_numpy(data_numpy.astype(np.float32))
        if x_0.ndim == 5: x_0 = x_0.squeeze(0)
        
        # Normalizzazione (Semplice min-max per ora, ma meglio se globale)
        x_0 = (x_0 - x_0.min()) / (x_0.max() - x_0.min() + 1e-8)

        # 2. Recupero Parametri per il CONTEXT (4 valori)
        try:
            row = self.catalog.loc[galaxy_id]
            params = []
            for col in self.feature_cols:
                c_min, c_max = self.stats[col]
                norm_val = (row[col] - c_min) / (c_max - c_min + 1e-8)
                params.append(norm_val)
            context_vector = torch.tensor(params, dtype=torch.float32)
        except KeyError:
            context_vector = torch.zeros(len(self.feature_cols))

        return {
            "x_0": x_0,              # Input per l'Encoder -> (1, 128, 128, 128)
            "context": context_vector # Input per l'MLP -> (4,)
        }
    

import torch.nn as nn

class CatalogueEmbedder(nn.Module):
    def __init__(self, input_dim=4, embed_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            # Espandiamo prima a 64 per estrarre correlazioni tra i parametri
            nn.Linear(input_dim, 64),
            nn.GELU(), # GELU è spesso preferita a ReLU nei Transformer/UNet moderne
            
            # Portiamo a 128
            nn.Linear(64, embed_dim),
            nn.LayerNorm(embed_dim) # Fondamentale per la stabilità della Cross-Attention
        )

    def forward(self, x):
        # x: [batch, 4] (es. [1, 4])
        emb = self.net(x) 
        # La UNet si aspetta [batch, sequence_length, context_dim]
        # In questo caso la sequence_length è 1 (una sorgente per patch)
        return emb.unsqueeze(1) # Risultato: [batch, 1, 128]