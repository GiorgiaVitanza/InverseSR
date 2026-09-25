# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pandas as pd
from glob import glob



def normalize_dynamic(data, norm_mode, stats=None):
    """
    Applica la normalizzazione ed evita la perdita dei min/max locali.
    Restituisce: (data_norm, patch_stats)
    """
    if stats is None: 
        stats = {}

    is_torch = isinstance(data, torch.Tensor)
    if not is_torch:
        data = torch.from_numpy(data).float()
    else:
        data = data.float()
        
    # --- CONTROLLO SICUREZZA PER TENSOR / ARRAY VUOTI ---
    numel = data.numel() if is_torch else data.size
    if numel == 0:
        return data, stats

    if norm_mode == 'global_sym':
        limit = stats.get('limit', 1.0)
        x_scaled = data / (limit + 1e-8)
        data_norm = (x_scaled + 1.0) / 2.0
        return torch.clamp(data_norm, 0.0, 1.0), {'limit': limit}

    elif norm_mode == 'global_robust':
        # Percentili 0.05% e 95% passati tramite il dizionario globale stats
        p_min = stats.get('p_min', -1.0559e-05)
        p_max = stats.get('p_max', 1.8725e-05)
        
        if p_max <= p_min:
            p_max = p_min + 1e-8

        data_norm = (data - p_min) / (p_max - p_min + 1e-8)
        patch_stats = {'p_min': p_min, 'p_max': p_max}

        return torch.clamp(data_norm, 0.0, 1.0), patch_stats

    elif norm_mode == 'global_arcsinh':
        # Normalizzazione Arcsinh usando i percentili 0.05% e 95%
        p_min = stats.get('p_min', -1.0559e-05)
        p_max = stats.get('p_max', 1.8725e-05)

        # Calcolo dinamico o recupero da stats dei limiti arcsinh
        p_min_arcsinh = stats.get('p_min_arcsinh', float(np.arcsinh(p_min)))
        p_max_arcsinh = stats.get('p_max_arcsinh', float(np.arcsinh(p_max)))

        if p_max_arcsinh <= p_min_arcsinh:
            p_max_arcsinh = p_min_arcsinh + 1e-8

        # Trasformazione logaritmica/arcsinh e scaling [0, 1]
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
        if is_torch:
            p_min = float(data.min())
            p_max = float(torch.quantile(data.float(), 0.998))
        else:
            p_min = float(np.min(data))
            p_max = float(np.percentile(data, 99.8))

        if p_max <= p_min:
            p_max = p_min + 1e-5

        data_norm = (data - p_min) / (p_max - p_min + 1e-8)
        patch_stats = {'p_min': p_min, 'p_max': p_max}

        return torch.clamp(data_norm, 0.0, 1.0), patch_stats

    elif norm_mode == 'zscore':
        mean = stats.get('mean', 0.0)
        std = stats.get('std', 1.0)
        data_norm = (data - mean) / (std + 1e-8)
        patch_stats = {'mean': mean, 'std': std}

        return torch.clamp(data_norm, -1.0, 1.0), patch_stats
        
    else:
        raise ValueError(f"Modalità '{norm_mode}' non supportata.")


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
        catalogue_path=None,  # <-- reso opzionale con default None
        in_channels=1,
        norm_mode="global_sym",
        num_samples_stats=100,
        mask_sigma=1.5,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.in_channels = in_channels
        self.norm_mode = norm_mode
        self.mask_sigma = mask_sigma
       

        # 1. SCANSIONE DIRETTORI
        all_paths = sorted(glob(os.path.join(data_dir, "*.npy")))
        self.patch_files = [os.path.basename(p) for p in all_paths]

        if len(self.patch_files) == 0:
            raise FileNotFoundError(
                f"Nessun file .npy trovato nella directory: {data_dir}"
            )

        # 2. CARICAMENTO E PULIZIA CATALOGO (Gestione opzionale)
        self.catalog_dict = {}
        self.feature_cols = ["hi_size", "line_flux_integral", "i", "w20"]
        self.stats_catalog = {}

        if catalogue_path and os.path.exists(catalogue_path):
            df_catalog = pd.read_csv(catalogue_path)
            df_catalog.columns = [
                c.lower().strip() for c in df_catalog.columns
            ]

            if "patch_id" in df_catalog.columns:
                df_catalog = df_catalog.drop_duplicates(
                    subset=["patch_id"], keep="first"
                )

                self.stats_catalog = {
                    col: (df_catalog[col].min(), df_catalog[col].max())
                    for col in self.feature_cols
                    if col in df_catalog.columns
                }

                self.catalog_dict = df_catalog.set_index("patch_id").to_dict(
                    orient="index"
                )
            else:
                print(
                    "Warning: Colonna 'patch_id' non trovata nel CSV. Il catalogo verrà ignorato."
                )
        else:
            print("Info: Nessun catalogo fornito o file non trovato. Si procede senza catalogo.")

        # 3. Calcolo dinamico statistiche
        self.dataset_stats = self._compute_dataset_statistics(
            num_samples_stats
        )

    def _compute_dataset_statistics(self, num_samples):
        print(
            f"Calcolo dinamico delle statistiche su {min(num_samples, len(self.patch_files))} file..."
        )
        sampled_files = np.random.choice(
            self.patch_files,
            size=min(num_samples, len(self.patch_files)),
            replace=False,
        )

        all_values = []
        max_absolute = 0.0

        for filename in sampled_files:
            path = os.path.join(self.data_dir, filename)
            data = np.load(path).astype(np.float32)
            max_absolute = max(max_absolute, np.max(np.abs(data)))
            all_values.append(data.ravel())

        all_values = np.concatenate(all_values)

        stats = {
            "limit": float(max_absolute),
            "mean": float(np.mean(all_values)),
            "std": float(np.std(all_values)),
        }

        print(
            f"Statistiche calcolate -> LIMIT: {stats['limit']:.4e}, MEAN: {stats['mean']:.4e}, STD: {stats['std']:.4e}"
        )
        return stats

    def __len__(self):
        return len(self.patch_files)

    def __getitem__(self, idx):
        filename = self.patch_files[idx]
        path = os.path.join(self.data_dir, filename)

        # Caricamento del file .npy
        data_numpy = np.load(path).astype(np.float32)

        if data_numpy.ndim == 4 and data_numpy.shape[0] == 1:
            data_numpy = data_numpy.squeeze(0)
        elif data_numpy.ndim == 5:
            data_numpy = data_numpy.squeeze()

        # Normalizzazione
        x_0, _ = normalize_dynamic(
            data_numpy, self.norm_mode, self.dataset_stats
        )

        if x_0.ndim == 3:
            x_0 = x_0.unsqueeze(0)  # Shape: (1, D, H, W)

        if self.in_channels == 3 and x_0.shape[0] == 1:
            x_0 = x_0.repeat(3, 1, 1, 1)

        D, H, W = x_0.shape[-3], x_0.shape[-2], x_0.shape[-1]

        # --- MASCHERA SPAZIALE 3D ---
        spatial_mask = torch.zeros((1, D, H, W), dtype=torch.float32)

        # Se il catalogo esiste ed è stato caricato per questo patch
        if self.catalog_dict and filename in self.catalog_dict:
            row = self.catalog_dict[filename]
            if all(k in row for k in ("rel_x", "rel_y", "rel_z")):
                rel_x, rel_y, rel_z = (
                    float(row["rel_x"]),
                    float(row["rel_y"]),
                    float(row["rel_z"]),
                )
                mask_np = create_3d_gaussian_mask(
                    (D, H, W),
                    x_c=rel_x,
                    y_c=rel_y,
                    z_c=rel_z,
                    sigma=self.mask_sigma,
                )
                spatial_mask = torch.from_numpy(mask_np).unsqueeze(0)

        

        # --- CONTEXT VECTOR ---
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
            # Fallback a valori neutri (0.5) se il catalogo non è presente
            params = [0.5] * len(self.feature_cols)

        context_vector = torch.tensor(params, dtype=torch.float32)

        return {
            "x_0": x_0,
            "spatial_mask": spatial_mask,
            "context": context_vector,
        }