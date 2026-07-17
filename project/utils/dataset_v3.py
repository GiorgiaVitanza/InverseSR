# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pandas as pd

def normalize_dynamic(data, norm_mode, stats={}):
    """
    Applica la normalizzazione usando le statistiche calcolate dinamicamente dal dataset.
    stats e' un dizionario contenente: 'limit', 'mean', 'std'
    """
    if norm_mode == 'global_sym':
        # Mappa [-LIMIT, LIMIT] -> [0, 1] con zero a 0.5 usando il LIMIT calcolato
        limit = stats['limit']
        x_scaled = data / (limit + 1e-8)
        data_norm = (x_scaled + 1.0) / 2.0
        try:
            return torch.from_numpy(np.clip(data_norm, 0, 1))
        except:
            return torch.clamp(data_norm, 0, 1)

    elif norm_mode == 'local':
        # Stretching basato sulla singola patch (rimane invariato perche' e' gia' locale)
        try:
            p_min = data.min()
            p_max = np.percentile(data, 99.8) 
        except:
            p_min, p_max = -1.47e-03, 1.52e-03
        data_norm = (data - p_min) / (p_max - p_min + 1e-8)
        try:
            return torch.from_numpy(np.clip(data_norm, 0, 1))
        except:
            return torch.clamp(data_norm, 0, 1)

    elif norm_mode == 'zscore':
        # Standardizzazione (media 0, deviazione 1) usando MEAN e STD calcolati dal dataset
        data_norm = (data - stats['mean']) / (stats['std'] + 1e-8)
        try:
            return torch.from_numpy(np.clip(data_norm, -1, 1))
        except:
            return torch.clamp(data_norm, -1, 1)
    else:
        raise ValueError(f"Modalita {norm_mode} non supportata.")


class RadioPatchDataset(Dataset):
    def __init__(self, data_dir, catalogue_path, in_channels, norm_mode='global_sym', num_samples_stats=100):
        """
        Args:
            data_dir: Path alle patch .npy
            catalogue_path: Path al CSV
            in_channels: Numero di canali 
            norm_mode: 'global_sym', 'local', o 'zscore'
            num_samples_stats: Numero di patch da scansionare per stimare la statistica (evita di caricarle tutte se sono troppe)
        """
        super().__init__()
        self.data_dir = data_dir
        self.in_channels = in_channels
        self.norm_mode = norm_mode
        
        # Caricamento catalogo
        self.catalog = pd.read_csv(catalogue_path)
        self.catalog.columns = [c.lower().strip() for c in self.catalog.columns]
        
        if 'patch_id' in self.catalog.columns:
            self.catalog = self.catalog.set_index("patch_id")
        else:
            print("Warning: Il catalogo deve contenere la colonna 'patch_id'")

        self.feature_cols = ['hi_size', 'line_flux_integral', 'i', 'w20']
        self.stats_catalog = {col: (self.catalog[col].min(), self.catalog[col].max()) 
                              for col in self.feature_cols}
        self.patch_files = self.catalog.index.tolist()

        # --- CALCOLO DINAMICO DELLE STATISTICHE DEL DATASET ---
        self.dataset_stats = self._compute_dataset_statistics(num_samples_stats)


    def _compute_dataset_statistics(self, num_samples):
        """Scansiona un sottoinsieme di file per calcolare il valore massimo assoluto, media e std."""
        print("Calcolo dinamico delle statistiche del dataset radio...")
        
        # Scegliamo un set di campioni casuali o limitati per velocizzare l'avvio del training
        sampled_files = np.random.choice(self.patch_files, size=min(num_samples, len(self.patch_files)), replace=False)
        
        all_values = []
        max_absolute = 0.0
        
        for filename in sampled_files:
            path = os.path.join(self.data_dir, filename)
            if os.path.exists(path):
                data = np.load(path).astype(np.float32)
                # Calcola il limite simmetrico assoluto (il picco piu' alto positivo o negativo)
                max_absolute = max(max_absolute, np.max(np.abs(data)))
                # Salviamo una versione appiattita per calcolare media e std globali
                all_values.append(data.ravel())
        
        # Concateniamo i pixel estratti per fare la statistica aggregata
        all_values = np.concatenate(all_values)
        
        stats = {
            'limit': float(max_absolute),
            'mean': float(np.mean(all_values)),
            'std': float(np.std(all_values))
        }
        
        print(f"Statistiche calcolate -> LIMIT (Max Assoluto): {stats['limit']:.4e}, MEAN: {stats['mean']:.4e}, STD: {stats['std']:.4e}")
        return stats


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

        # Passiamo il dizionario self.dataset_stats calcolato dinamicamente nell'__init__
        x_0 = normalize_dynamic(data_numpy, self.norm_mode, self.dataset_stats)
        
        if x_0.ndim == 3: 
            x_0 = x_0.unsqueeze(0) 

        if self.in_channels == 3 and x_0.shape[0] == 1:
            x_0 = x_0.repeat(3, 1, 1, 1)

        # Context 
        try:
            row = self.catalog.loc[filename]
            if isinstance(row, pd.DataFrame): 
                row = row.iloc[0] 
            
            params = []
            for col in self.feature_cols:
                c_min, c_max = self.stats_catalog[col]
                norm_val = (row[col] - c_min) / (c_max - c_min + 1e-8)
                params.append(np.clip(norm_val, 0, 1))
            
            context_vector = torch.tensor(params, dtype=torch.float32)
        except KeyError:
            context_vector = torch.full((len(self.feature_cols),), 0.5)

        return {"x_0": x_0, "context": context_vector}