import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse


import os, psutil, resource
import torch


def log_memory_usage(device: str, metrics: dict):
    if "cuda" in str(device) and torch.cuda.is_available():
        # --- METRICHE GPU (VRAM) ---
        torch.cuda.synchronize()
        metrics["Peak_VRAM_Allocated_MB"] = float(
            torch.cuda.max_memory_allocated() / (1024**2)
        )
        metrics["Peak_VRAM_Reserved_MB"] = float(
            torch.cuda.max_memory_reserved() / (1024**2)
        )

    else:
        # --- METRICHE CPU (RAM) ---
        process = psutil.Process(os.getpid())

        # Peak RAM (RSS - Resident Set Size) dall'inizio dell'esecuzione del processo
        # Nota: ru_maxrss su Linux restituisce Kilobyte
        import resource

        peak_ram_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        metrics["Peak_CPU_RAM_MB"] = float(peak_ram_kb / 1024)

        # RAM attualmente occupata dal processo Python
        current_ram_bytes = process.memory_info().rss
        metrics["Current_CPU_RAM_MB"] = float(current_ram_bytes / (1024**2))

def compute_3d_power_spectrum(volume: np.ndarray):
    """
    Calcola lo Spettro di Potenza 3D (Radial Power Spectrum) isotropico per volumi sfericamente simmetrici.
    Valuta se la SR conserva le frequenze spaziali (dettagli fini vs grandi strutture).
    """
    # 3D FFT e Fast Shift al centro
    fft3d = np.fft.fftn(volume)
    fft3d_shifted = np.fft.fftshift(fft3d)
    power_spectrum = np.abs(fft3d_shifted) ** 2

    nz, ny, nx = volume.shape
    z, y, x = np.indices((nz, ny, nx))
    center = np.array([nz // 2, ny // 2, nx // 2])
    
    # Calcolo distanza dal centro delle frequenze k
    r = np.sqrt((z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2).astype(int)

    # Binning radiale
    tbin = np.bincount(r.ravel(), power_spectrum.ravel())
    nr = np.bincount(r.ravel())
    
    # Evita divisioni per zero nei bin vuoti
    radial_profile = np.divide(tbin, nr, out=np.zeros_like(tbin, dtype=float), where=nr != 0)
    return radial_profile


def run_astrophysical_benchmarks(
    target_phys: np.ndarray, 
    synth_phys: np.ndarray, 
    target_corrupted_phys: np.ndarray = None,
    latent_tensor: torch.Tensor = None,
    device: str = "cuda"
) -> dict:
    """
    Esegue la suite completa di test fisici, visivi, di compressione e memoria (CPU/GPU).
    """
    metrics = {}

    # ==========================================
    # 1. METRICHE VISIVE E STRUTTURALI
    # ==========================================
    data_range = max(target_phys.max(), synth_phys.max()) - min(target_phys.min(), synth_phys.min())
    if data_range == 0:
        data_range = 1e-5

    metrics["PSNR"] = float(psnr(target_phys, synth_phys, data_range=data_range))
    metrics["MSE"] = float(mse(target_phys, synth_phys))
    metrics["SSIM_3D"] = float(ssim(synth_phys, target_phys, data_range=data_range))

    # ==========================================
    # 2. FEDELTÀ E CONSERVAZIONE FISICA
    # ==========================================
    flux_target = np.sum(target_phys)
    flux_synth = np.sum(synth_phys)
    
    metrics["Target_Total_Flux"] = float(flux_target)
    metrics["Synth_Total_Flux"] = float(flux_synth)
    metrics["Flux_Conservation_Error_%"] = float(np.abs((flux_synth - flux_target) / (flux_target + 1e-12)) * 100)

    hist_target, bin_edges = np.histogram(target_phys, bins=50, density=True)
    hist_synth, _ = np.histogram(synth_phys, bins=bin_edges, density=True)
    
    p = hist_target + 1e-12
    q = hist_synth + 1e-12
    metrics["PDF_KL_Divergence"] = float(np.sum(p * np.log(p / q)))

    ps_target = compute_3d_power_spectrum(target_phys)
    ps_synth = compute_3d_power_spectrum(synth_phys)
    metrics["Power_Spectrum_Relative_Error_%"] = float(np.mean(np.abs(ps_synth - ps_target) / (ps_target + 1e-12)) * 100)

    # ==========================================
    # 3. METRICHE DI COMPRESSIONE ED EFFICIENZA
    # ==========================================
    bytes_uncompressed = target_phys.size * 4
    
    if latent_tensor is not None:
        bytes_compressed = latent_tensor.element_size() * latent_tensor.nelement()
        metrics["Compression_Ratio"] = float(bytes_uncompressed / bytes_compressed)
        metrics["Space_Saving_%"] = float((1 - (bytes_compressed / bytes_uncompressed)) * 100)

    # ==========================================
    # 4. BENCHMARK MEMORIA (GPU vs CPU)
    # ==========================================
    if "cuda" in str(device) and torch.cuda.is_available():
        torch.cuda.synchronize()
        metrics["Peak_VRAM_Allocated_MB"] = float(torch.cuda.max_memory_allocated() / (1024 ** 2))
        metrics["Peak_VRAM_Reserved_MB"] = float(torch.cuda.max_memory_reserved() / (1024 ** 2))
    else:
        # Gestione RAM su CPU (compatibile con i nodi Linux/Slurm di Leonardo)
        process = psutil.Process(os.getpid())
        peak_ram_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        
        import platform
        scale = 1024 * 1024 if platform.system() == "Darwin" else 1024
        
        metrics["Peak_CPU_RAM_MB"] = float(peak_ram_kb / scale)
        metrics["Current_CPU_RAM_MB"] = float(process.memory_info().rss / (1024 ** 2))

    return metrics