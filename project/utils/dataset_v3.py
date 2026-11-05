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

    # --------------------------------------------------------------------------
    # 1. INTEGRAZIONE FILTRO OUTLIER + RISCALAMENTO (minmax_clip & clip)
    # --------------------------------------------------------------------------
    if norm_mode in ['minmax_clip', 'clip']:
        # Estrazione robusta delle soglie (gestisce JSON sia flat che annidati)
        if "filtering_bounds" in stats:
            lower_b = stats["filtering_bounds"]["lower_bound"]
            upper_b = stats["filtering_bounds"]["upper_bound"]
        else:
            lower_b = stats.get("lower_bound", float(data.min()))
            upper_b = stats.get("upper_bound", float(data.max()))

        # Passaggio 1: Applicazione Clipping per eliminare gli outlier
        data_clipped = torch.clamp(data, min=lower_b, max=upper_b)

        if norm_mode == 'clip':
            # Solo clipping nei limiti reali [lower_b, upper_b]
            return data_clipped, {'lower_bound': lower_b, 'upper_bound': upper_b}
        
        elif norm_mode == 'minmax_clip':
            # Passaggio 2: Riscalamento lineare dei dati filtrati nell'intervallo [0.0, 1.0]
            data_norm = (data_clipped - lower_b) / (upper_b - lower_b + 1e-8)
            return data_norm, {'lower_bound': lower_b, 'upper_bound': upper_b}

    # --------------------------------------------------------------------------
    # 2. ALTRE MODALITÀ DI NORMALIZZAZIONE
    # --------------------------------------------------------------------------
    
    
    elif norm_mode == 'global_robust':
        # 1. Trova i percentili dal dataset (o calcolali sul dato locale)
        # Questo elimina gli outlier estremi prima di scalare tra 0 e 1
        p_low = stats.get('p1', np.percentile(data, 1)) if stats else np.percentile(data, 1)
        p_high = stats.get('p99', np.percentile(data, 99)) if stats else np.percentile(data, 99)
        
        # Clip per rimuovere picchi anomali
        data_clipped = np.clip(data, p_low, p_high)
        
        # Scala l'intervallo compattato tra 0 e 1
        norm_data = (data_clipped - p_low) / (p_high - p_low + 1e-8)
        return norm_data, {"p_low": p_low, "p_high": p_high}

    elif norm_mode == 'log_minmax':
        # Utile se i dati coprono diversi ordini di grandezza (es. segnali di sismica o microtomografia)
        data_pos = np.maximum(data, 0)
        log_data = np.log1p(data_pos)  # log(1 + x) evita log(0)
        
        vmin, vmax = log_data.min(), log_data.max()
        norm_data = (log_data - vmin) / (vmax - vmin + 1e-8)
        return norm_data, {"vmin": vmin, "vmax": vmax}

    elif norm_mode == 'zscore_robust':
        # Standardizzazione basata su Mediana e IQR (più robusta di Media e Std)
        med = np.median(data)
        q25, q75 = np.percentile(data, [25, 75])
        iqr = q75 - q25
        norm_data = (data - med) / (iqr + 1e-8)
        return norm_data, {"median": med, "iqr": iqr}
    elif norm_mode in ['global_1', 'global_2']:
        raw_stats = stats.get("flux_statistics_raw", {}).get("percentiles", stats)
        p_min_key = 'p1' if norm_mode == 'global_1' else 'p2'
        p_max_key = 'p99' if norm_mode == 'global_1' else 'p98'

        p_min = raw_stats.get(p_min_key, stats.get(p_min_key, 0.0))
        p_max = raw_stats.get(p_max_key, stats.get(p_max_key, 1.0))

        if p_max <= p_min:
            p_max = p_min + 1e-8

        data_norm = (data - p_min) / (p_max - p_min + 1e-8)
        return torch.clamp(data_norm, 0.0, 1.0), {'p_min': p_min, 'p_max': p_max}

    elif norm_mode == 'global_arcsinh':
        raw_stats = stats.get("flux_statistics_raw", {}).get("percentiles", stats)
        p_min = raw_stats.get('p5', stats.get('p5', 0.0))
        p_max = raw_stats.get('p95', stats.get('p95', 1.0))
        p_min_arcsinh = raw_stats.get('p5_arcsinh', stats.get('p5_arcsinh', 0.0))
        p_max_arcsinh = raw_stats.get('p95_arcsinh', stats.get('p95_arcsinh', 1.0))

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
        return torch.clamp(data_norm, 0.0, 1.0), {'p_min': p_min, 'p_max': p_max}

    else:
        return data, {}


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
        cached_stats=None,
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

        # 2. CARICAMENTO STATISTICHE (Cerca sia dataset_filter_stats.json sia dataset_stat.json)
        self.dataset_stats = {}
        if cached_stats is not None:
            if isinstance(cached_stats, dict):
                self.dataset_stats = cached_stats
                print("Statistiche caricate direttamente dal parametro cached_stats.")
            elif isinstance(cached_stats, str) and os.path.exists(cached_stats):
                with open(cached_stats, "r") as f:
                    self.dataset_stats = json.load(f)
                print(f"Statistiche caricate dal file indicato in cached_stats: {cached_stats}")
        else:
            candidate_files = ["dataset_filter_stats.json", "dataset_stat.json"]
            candidate_dirs = [
                data_dir,
                os.path.dirname(data_dir),
                os.path.dirname(os.path.dirname(data_dir))
            ]

            found_json = None
            for d in candidate_dirs:
                for f_name in candidate_files:
                    full_path = os.path.join(d, f_name)
                    if os.path.exists(full_path):
                        found_json = full_path
                        break
                if found_json:
                    break

            if found_json:
                with open(found_json, "r") as f:
                    self.dataset_stats = json.load(f)
                print(f"Statistiche del dataset caricate con successo da: {found_json}")
            else:
                print(f"[Warning] Nessun file di statistiche trovato. Verrà usato un dizionario vuoto.")

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

        # Normalizzazione + Filtering
        x_0, _ = normalize_dynamic(data_numpy, self.norm_mode, self.dataset_stats)

        if x_0.ndim == 3:
            x_0 = x_0.unsqueeze(0)  # Shape: (1, D, H, W)

        # Usa .expand() per risparmiare memoria RAM
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