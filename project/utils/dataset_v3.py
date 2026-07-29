# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pandas as pd


def normalize_dynamic(data, norm_mode, stats={}):
    """
    Applica la normalizzazione usando le statistiche calcolate dinamicamente dal dataset.
    """
    if norm_mode == 'global_sym':
        limit = stats['limit']
        x_scaled = data / (limit + 1e-8)
        data_norm = (x_scaled + 1.0) / 2.0
        try:
            return torch.from_numpy(np.clip(data_norm, 0, 1))
        except:
            return torch.clamp(data_norm, 0, 1)

    elif norm_mode == 'local':
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
        data_norm = (data - stats['mean']) / (stats['std'] + 1e-8)
        try:
            return torch.from_numpy(np.clip(data_norm, -1, 1))
        except:
            return torch.clamp(data_norm, -1, 1)
    else:
        raise ValueError(f"Modalità {norm_mode} non supportata.")


def create_3d_gaussian_mask(shape, x_c, y_c, z_c, sigma=1.5):
    """
    Genera un volume 3D (D, H, W) contenente una Gaussiana centrata in (z_c, y_c, x_c).
    Fornisce un condizionamento spaziale 'soft' alla rete neurale.
    """
    D, H, W = shape
    z_grid, y_grid, x_grid = np.ogrid[:D, :H, :W]
    
    # Distanza euclidea 3D dal centro sorgente
    dist_sq = (x_grid - x_c)**2 + (y_grid - y_c)**2 + (z_grid - z_c)**2
    gaussian = np.exp(-dist_sq / (2 * (sigma**2)))
    
    return gaussian.astype(np.float32)


class RadioPatchDataset(Dataset):
    def __init__(
        self, 
        data_dir, 
        catalogue_path, 
        in_channels=1, 
        norm_mode='global_sym', 
        num_samples_stats=100,
        mask_sigma=1.5
    ):
        super().__init__()
        self.data_dir = data_dir
        self.in_channels = in_channels
        self.norm_mode = norm_mode
        self.mask_sigma = mask_sigma
        
        # 1. Caricamento e pulizia Catalogo
        self.catalog = pd.read_csv(catalogue_path)
        self.catalog.columns = [c.lower().strip() for c in self.catalog.columns]
        
        if 'patch_id' in self.catalog.columns:
            self.catalog = self.catalog.set_index("patch_id")
        else:
            print("Warning: Manca la colonna 'patch_id' come indice!")

        # Feature per il context scalare fisso
        self.feature_cols = ['hi_size', 'line_flux_integral', 'i', 'w20']
        
        # Calcolo min/max per la normalizzazione dei parametri del catalogo
        self.stats_catalog = {
            col: (self.catalog[col].min(), self.catalog[col].max()) 
            for col in self.feature_cols if col in self.catalog.columns
        }
        self.patch_files = self.catalog.index.tolist()

        # 2. Calcolo dinamico statistiche sulle patch .npy
        self.dataset_stats = self._compute_dataset_statistics(num_samples_stats)

    def _compute_dataset_statistics(self, num_samples):
        print("Calcolo dinamico delle statistiche del dataset radio...")
        sampled_files = np.random.choice(
            self.patch_files, size=min(num_samples, len(self.patch_files)), replace=False
        )
        
        all_values = []
        max_absolute = 0.0
        
        for filename in sampled_files:
            path = os.path.join(self.data_dir, filename)
            if os.path.exists(path):
                data = np.load(path).astype(np.float32)
                max_absolute = max(max_absolute, np.max(np.abs(data)))
                all_values.append(data.ravel())
        
        all_values = np.concatenate(all_values)
        
        stats = {
            'limit': float(max_absolute),
            'mean': float(np.mean(all_values)),
            'std': float(np.std(all_values))
        }
        
        print(f"Statistiche calcolate -> LIMIT: {stats['limit']:.4e}, MEAN: {stats['mean']:.4e}, STD: {stats['std']:.4e}")
        return stats

    def __len__(self):
        return len(self.patch_files)

    def __getitem__(self, idx):
        filename = self.patch_files[idx]
        path = os.path.join(self.data_dir, filename)
        
        # Caricamento patch .npy
        data_numpy = np.load(path).astype(np.float32)

        if data_numpy.ndim == 4 and data_numpy.shape[0] == 1:
            data_numpy = data_numpy.squeeze(0)
        elif data_numpy.ndim == 5:
            data_numpy = data_numpy.squeeze()

        # Normalizzazione Cubo
        x_0 = normalize_dynamic(data_numpy, self.norm_mode, self.dataset_stats)
        
        if x_0.ndim == 3: 
            x_0 = x_0.unsqueeze(0)  # Shape finale: (1, D, H, W)

        if self.in_channels == 3 and x_0.shape[0] == 1:
            x_0 = x_0.repeat(3, 1, 1, 1)

        # Dimensioni effettive della patch (es. D=16, H=128, W=128)
        D, H, W = x_0.shape[-3], x_0.shape[-2], x_0.shape[-1]

        # --- GENERAZIONE MASCHERA SPATIALE 3D ---
        spatial_mask = torch.zeros((1, D, H, W), dtype=torch.float32)
        
        try:
            row = self.catalog.loc[filename]
            if isinstance(row, pd.DataFrame): 
                row = row.iloc[0] 

            # Estrazione coordinate dal tuo CSV
            rel_x = float(row['rel_x'])
            rel_y = float(row['rel_y'])
            rel_z = float(row['rel_z'])

            # Generiamo la Gaussiana 3D per ancorare la sorgente alle coordinate esatte
            mask_np = create_3d_gaussian_mask((D, H, W), x_c=rel_x, y_c=rel_y, z_c=rel_z, sigma=self.mask_sigma)
            spatial_mask = torch.from_numpy(mask_np).unsqueeze(0)  # Shape: (1, D, H, W)

        except (KeyError, ValueError):
            pass

        # --- ESTRAZIONE PARAMETRI SCALARI (Context Vector) ---
        try:
            params = []
            for col in self.feature_cols:
                if col in row:
                    c_min, c_max = self.stats_catalog[col]
                    norm_val = (row[col] - c_min) / (c_max - c_min + 1e-8)
                    params.append(np.clip(norm_val, 0, 1))
                else:
                    params.append(0.5)
            
            context_vector = torch.tensor(params, dtype=torch.float32)
        except NameError:
            context_vector = torch.full((len(self.feature_cols),), 0.5)

        return {
            "x_0": x_0,                       # Shape: (1, D, H, W)
            "spatial_mask": spatial_mask,     # Shape: (1, D, H, W) -> LA MASCHERA DI POSIZIONE
            "context": context_vector,         # Shape: (4,)
        }