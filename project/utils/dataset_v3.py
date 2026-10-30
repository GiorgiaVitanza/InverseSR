# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pandas as pd
from glob import glob
from utils.dataset_stats import compute_dataset_stats
import json

def normalize_dynamic(data, norm_mode, stats=None):
    if stats is None: 
        stats = {}

    if not isinstance(data, torch.Tensor):
        data = torch.from_numpy(data).float()
    else:
        data = data.float()
        
    if data.numel() == 0:
        return data, stats

    if norm_mode == 'global_robust':
        p_min = stats['p5']
        p_max = stats['p95']
        
        if p_max <= p_min:
            p_max = p_min + 1e-8

        data_norm = (data - p_min) / (p_max - p_min + 1e-8)
        patch_stats = {'p_min': p_min, 'p_max': p_max}

        return torch.clamp(data_norm, 0.0, 1.0), patch_stats
    elif norm_mode == 'global_1':
                p_min = stats['p1']
                p_max = stats['p99']
                
                if p_max <= p_min:
                    p_max = p_min + 1e-8
        
                data_norm = (data - p_min) / (p_max - p_min + 1e-8)
                patch_stats = {'p_min': p_min, 'p_max': p_max}
        
                return torch.clamp(data_norm, 0.0, 1.0), patch_stats
        
    elif norm_mode == 'global_2':
            p_min = stats['p2']
            p_max = stats['p98']
            
            if p_max <= p_min:
                p_max = p_min + 1e-8
    
            data_norm = (data - p_min) / (p_max - p_min + 1e-8)
            patch_stats = {'p_min': p_min, 'p_max': p_max}
    
            return torch.clamp(data_norm, 0.0, 1.0), patch_stats
    

    elif norm_mode == 'global_arcsinh':
        p_min = stats['p5']
        p_max = stats['p95']

        p_min_arcsinh = stats['p5_arcsinh']
        p_max_arcsinh = stats['p95_arcsinh']

        if p_max_arcsinh <= p_min_arcsinh:
            p_max_arcsinh = p_min_arcsinh + 1e-8

        data_arcsinh = torch.arcsinh(data)
        data_norm = (data_arcsinh - p_min_arcsinh) / (p_max_arcsinh - p_min_arcsinh + 1e-8)

        patch_stats = {
            'p_min': p_min,
            'p_max': p_max,
            'p_min_arcsinh': p_min_arcsinh,
            'p_max_arcsinh': p_max_arcsinh,
        }

        return torch.clamp(data_norm, 0.0, 1.0), patch_stats

    elif norm_mode == 'local':
        p_min = float(data.min())
        p_max = float(torch.quantile(data, 0.99))

        if p_max <= p_min:
            p_max = p_min + 1e-5

        data_norm = (data - p_min) / (p_max - p_min + 1e-8)
        patch_stats = {'p_min': p_min, 'p_max': p_max}

        return torch.clamp(data_norm, 0.0, 1.0), patch_stats

    elif norm_mode == 'zscore':
        mean = stats['mean']
        std = stats['std']
        data_norm = (data - mean) / (std + 1e-8)
        patch_stats = {'mean': mean, 'std': std}

        return torch.clamp(data_norm, -1.0, 1.0), patch_stats
        
    else:
        raise ValueError(f"Modalità '{norm_mode}' non supportata.")


def create_3d_gaussian_mask(shape, x_c, y_c, z_c, sigma=1.5):
    D, H, W = shape
    z_grid, y_grid, x_grid = np.ogrid[:D, :H, :W]
    dist_sq = (x_grid - x_c)**2 + (y_grid - y_c)**2 + (z_grid - z_c)**2
    gaussian = np.exp(-dist_sq / (2 * (sigma**2)))
    return gaussian.astype(np.float32)


class RadioPatchDataset(Dataset):
    def __init__(
        self,
        data_dir,
        catalogue_path=None,
        in_channels=1,
        norm_mode="global",
        mask_sigma=1.5,
        cached_stats=None,  # Consente di passare direttamente un dict o percorso
    ):
        super().__init__()
        self.data_dir = data_dir
        self.in_channels = in_channels
        self.norm_mode = norm_mode
        self.mask_sigma = mask_sigma

        all_paths = sorted(glob(os.path.join(data_dir, "*.npy")))
        self.patch_files = [os.path.basename(p) for p in all_paths]

        if len(self.patch_files) == 0:
            raise FileNotFoundError(f"Nessun file .npy trovato in: {data_dir}")

        # 1. CARICAMENTO CATALOGO
        self.catalog_dict = {}
        self.feature_cols = ["hi_size", "line_flux_integral", "i", "w20"]
        self.stats_catalog = {}

        if catalogue_path and os.path.exists(catalogue_path):
            df_catalog = pd.read_csv(catalogue_path)
            df_catalog.columns = [c.lower().strip() for c in df_catalog.columns]

            if "patch_id" in df_catalog.columns:
                df_catalog = df_catalog.drop_duplicates(subset=["patch_id"], keep="first")
                self.stats_catalog = {
                    col: (df_catalog[col].min(), df_catalog[col].max())
                    for col in self.feature_cols
                    if col in df_catalog.columns
                }
                self.catalog_dict = df_catalog.set_index("patch_id").to_dict(orient="index")

        # 2. CARICAMENTO STATISTICHE (ROBUSTO)
        if cached_stats is not None:
            if isinstance(cached_stats, dict):
                self.dataset_stats = cached_stats
                print("Statistiche caricate direttamente dal parametro cached_stats.")
            elif isinstance(cached_stats, str) and os.path.exists(cached_stats):
                with open(cached_stats, "r") as f:
                    self.dataset_stats = json.load(f)
                print(f"Statistiche caricate dal file indicato in cached_stats: {cached_stats}")
        else:
            # Lista dei percorsi possibili dove cercare il file JSON
            candidate_paths = [
                os.path.join(data_dir, "dataset_stat.json"),                        # dentro la cartella corrente
                os.path.join(os.path.dirname(data_dir), "dataset_stat.json"),       # nella cartella padre (es. train/ o val/)
                os.path.join(os.path.dirname(os.path.dirname(data_dir)), "dataset_stat.json") # due livelli sopra
            ]

            found_json = None
            for path in candidate_paths:
                if os.path.exists(path):
                    found_json = path
                    break

            if found_json:
                with open(found_json, "r") as f:
                    self.dataset_stats = json.load(f)
                print(f"Statistiche del dataset caricate con successo da: {found_json}")
            else:
                print(f"[Warning] Nessun file 'dataset_stat.json' trovato nei percorsi cercati. Esecuzione calcolo dinamico...")
                self.dataset_stats = compute_dataset_stats(data_dir)

    def __len__(self):
        return len(self.patch_files)

    def __getitem__(self, idx):
        filename = self.patch_files[idx]
        path = os.path.join(self.data_dir, filename)

        # Caricamento singolo file
        data_numpy = np.load(path).astype(np.float32)

        if data_numpy.ndim == 4 and data_numpy.shape[0] == 1:
            data_numpy = data_numpy.squeeze(0)
        elif data_numpy.ndim == 5:
            data_numpy = data_numpy.squeeze()

        # Normalizzazione
        x_0, _ = normalize_dynamic(data_numpy, self.norm_mode, self.dataset_stats)

        if x_0.ndim == 3:
            x_0 = x_0.unsqueeze(0)  # Shape: (1, D, H, W)

        # Usa .expand() al posto di .repeat() per evitare di triplicare l'allocazione RAM
        if self.in_channels == 3 and x_0.shape[0] == 1:
            x_0 = x_0.expand(3, -1, -1, -1)

        D, H, W = x_0.shape[-3], x_0.shape[-2], x_0.shape[-1]

        # Maschera Spaziale 3D
        spatial_mask = torch.zeros((1, D, H, W), dtype=torch.float32)
        if self.catalog_dict and filename in self.catalog_dict:
            row = self.catalog_dict[filename]
            if all(k in row for k in ("rel_x", "rel_y", "rel_z")):
                mask_np = create_3d_gaussian_mask(
                    (D, H, W),
                    x_c=float(row["rel_x"]),
                    y_c=float(row["rel_y"]),
                    z_c=float(row["rel_z"]),
                    sigma=self.mask_sigma,
                )
                spatial_mask = torch.from_numpy(mask_np).unsqueeze(0)

        # Context Vector
        params = []
        if self.catalog_dict and filename in self.catalog_dict:
            row = self.catalog_dict[filename]
            for col in self.feature_cols:
                if col in row and col in self.stats_catalog:
                    c_min, c_max = self.stats_catalog[col]
                    norm_val = (row[col] - c_min) / (c_max - c_min + 1e-8)
                    params.append(np.clip(norm_val, 0, 1))
                else:
                    params.append(0.5)
        else:
            params = [0.5] * len(self.feature_cols)

        context_vector = torch.tensor(params, dtype=torch.float32)

        
        return {
            "x_0": x_0,
            "spatial_mask": spatial_mask,
            "context": context_vector,
            "stats": self.dataset_stats,
        }