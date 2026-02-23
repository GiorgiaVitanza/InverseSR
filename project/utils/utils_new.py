import os
import random
from argparse import Namespace
from pathlib import Path
from typing import Tuple, Dict, List, Any

import mlflow
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from monai.transforms import apply_transform
from tqdm import tqdm

# Assicurati che questi import puntino ai tuoi moduli corretti
from models.ddim import DDIMSampler
from models.aekl_no_attention import OnlyDecoder
from models.ddpm_v2_conditioned import DDPM
from models.BRGM.forward_models import (
    ForwardDownsample,
    ForwardFillMask,
    ForwardAbstract,
)
from utils.transorms import get_preprocessing
from utils.const import (
    INPUT_FOLDER_PATCHES,
    MASK_FOLDER,
    PRETRAINED_MODEL_VAE_PATH,
    PRETRAINED_MODEL_DDPM_PATH,
    PRETRAINED_MODEL_VGG_PATH,
    OUTPUT_FOLDER,
    LATENT_SHAPE, # Assicurati che in const.py questo sia corretto per i tuoi dati (es. [1, 3, 16, 16, 16])
)

# --- UTILS ---

def transform_img(img_path: Path, device: torch.device) -> Any:
    """Applica le trasformazioni MONAI all'immagine."""
    data = {"image": img_path}
    # get_preprocessing deve essere quello adattato per FITS che abbiamo fatto prima
    data = apply_transform(get_preprocessing(device), data)
    return data["image"]

def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# --- LOADING DATA & MODELS ---

def load_target_image(hparams: Namespace, device: torch.device) -> torch.Tensor:
    """
    Carica l'immagine target (Ground Truth) da file FITS o da patch .npy.
    """
    if hparams.data_format == "npy":
        # Cerca il patch npy (es. patch_001.npy)
        potential_files = list(INPUT_FOLDER_PATCHES.glob(f"*{hparams.object_id}*.npy"))
        
        if not potential_files:
            raise FileNotFoundError(f"Nessun patch .npy trovato per ID {hparams.object_id} in {INPUT_FOLDER_PATCHES}")
        
        img_path = potential_files[0]
        # Carichiamo il file numpy
        data = np.load(img_path).astype(np.float32)
        
        # Converte in tensor
        img_tensor = torch.from_numpy(data).to(device)
        
        # Gestione dimensioni: 
        # Se il patch è (D, H, W), lo portiamo a (C, D, H, W) aggiungendo il canale
        if img_tensor.ndim == 3:
            img_tensor = img_tensor.unsqueeze(0)
            
    elif hparams.data_format == "fits":
        potential_files = list(INPUT_FOLDER_PATCHES.glob(f"*{hparams.object_id}*.fits"))
        if not potential_files:
            raise FileNotFoundError(f"Nessun file FITS trovato per ID {hparams.object_id}")
        
        img_path = potential_files[0]
        img_tensor = transform_img(img_path, device=device)
        # transform_img dovrebbe già restituire (C, D, H, W)
        
    else:
        raise ValueError(f"Formato {hparams.data_format} non supportato.")

    # Aggiunge la dimensione Batch: -> (1, C, D, H, W)
    if img_tensor.ndim == 4:
        img_tensor = img_tensor.unsqueeze(0)
        
    return img_tensor


def load_pre_trained_model(device: torch.device) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """Carica VAE (Decoder) e DDPM tramite MLFlow."""
    print(f"Caricamento modelli da:\nVAE: {PRETRAINED_MODEL_VAE_PATH}\nDDPM: {PRETRAINED_MODEL_DDPM_PATH}")
    
    decoder = mlflow.pytorch.load_model(str(PRETRAINED_MODEL_VAE_PATH), map_location=device)
    ddpm = mlflow.pytorch.load_model(str(PRETRAINED_MODEL_DDPM_PATH), map_location=device)
    
    decoder.eval().to(device).requires_grad_(False)
    ddpm.eval().to(device).requires_grad_(False)
      
    return ddpm, decoder

def create_corruption_function(hparams: Namespace, device: torch.device) -> ForwardAbstract:
    """Definisce come 'rovinare' l'immagine (per i test di ricostruzione)."""
    if hparams.corruption == "downsample":
        forward = ForwardDownsample(factor=hparams.downsample_factor)
    elif hparams.corruption == "mask":
        mask_path = MASK_FOLDER / f"{hparams.mask_id}.npy"
        if not mask_path.exists():
             raise FileNotFoundError(f"Maschera non trovata: {mask_path}")
        mask = np.load(mask_path)
        forward = ForwardFillMask(mask=mask, device=device)
    else:
        # Nessuna corruzione (Identity)
        forward = ForwardFillMask(device=device)
    return forward

# --- CONDITIONING UTILS ---

def setup_noise_inputs(device: torch.device, hparams: Namespace) -> Tuple[torch.Tensor, torch.Tensor]:
    # Creiamo un vettore di 8 parametri (es. tutti a 0.5 o valori medi del catalogo)
    cond_full = torch.full((1, 8), 0.5, device=device, dtype=torch.float32)
    
    # Se vuoi ottimizzare i primi due (es. Freq e Flux sono i primi due nel tuo dataset)
    # li rendiamo parte del grafo di calcolo
    cond_full.requires_grad_(True)

    # Rumore latente (z_channels=3, resolution=32)
    latent_shape = (1, 3, 32, 32, 32)
    latent_variable = torch.randn(latent_shape, device=device, requires_grad=True)
    
    return cond_full, latent_variable

def _prepare_conditioning_dict(cond):
    cond_tmp = cond.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # shape: [1, 4, 1, 1, 1]
    cond_crossatten = (
        cond_tmp.squeeze(-1).squeeze(-1).squeeze(-1).unsqueeze(1)
    )  # shape: [1, 1, 4]
    cond_concat = torch.tile(cond_tmp, (32, 32, 32))
    conditioning = {
        "c_concat": [cond_concat],
        "c_crossattn": [cond_crossatten],
    }
       

def sampling_from_ddim(
    ddim: DDIMSampler,
    latent_variable: torch.Tensor,
    decoder: OnlyDecoder,
    cond: torch.Tensor,
    hparams: Namespace,
) -> torch.Tensor:
    
    # Prepara il dizionario di condizionamento dinamicamente
    conditioning = _prepare_conditioning_dict(cond)

    print(f"Start DDIM Sampling ({hparams.ddim_num_timesteps} steps)...")
    latent_vectors, _ = ddim.sample(
        S=hparams.ddim_num_timesteps,
        conditioning=conditioning,
        batch_size=1,
        shape=list(LATENT_SHAPE[1:]), # Esclude dimensione Batch
        first_img=latent_variable,
        eta=hparams.ddim_eta,
        verbose=False,
    )
   
    
    # --- DECODING (Memory Optimized) ---
    # Utile per datacube grandi
    import gc
    del conditioning
    gc.collect()
    torch.cuda.empty_cache()

    if hasattr(latent_vectors, "as_tensor"):
        latent_vectors = latent_vectors.as_tensor()

    # Passaggio a FP16 per il decoding (risparmia VRAM)
    decoder.half()
    latent_vectors = latent_vectors.half()

    with torch.no_grad():
        astro_img = decoder.reconstruct_ldm_outputs(latent_vectors)
    
    return astro_img.float() # Torna a FP32


# --- PERCEPTUAL LOSS UTILS (VGG) ---

def load_vgg_perceptual(hparams: Namespace, target: torch.Tensor, device: torch.device) -> Tuple[Any, torch.Tensor]:
    """Carica VGG16 per calcolare la Perceptual Loss."""
    # Nota: VGG è 2D. Se il target è 3D, dovremo fare slicing.
    with open(PRETRAINED_MODEL_VGG_PATH, "rb") as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    target_features = getVggFeatures(hparams, target, vgg16)
    return vgg16, target_features

def getVggFeatures(hparams: Namespace, img: torch.Tensor, vgg16: Any) -> torch.Tensor:
    """
    Estrae le feature VGG. 
    Poiché VGG vuole input 2D RGB (3 canali), dobbiamo fare slicing del cubo 3D 
    e duplicare il canale singolo (grayscale) in 3.
    """
    # img shape: [B, C, D, H, W]
    
    # 1. Resize opzionale se l'immagine corrotta è downsampled
    if hparams.corruption == "downsample":
        # Nota: interpolate vuole float, dimensioni target fisse o calcolate
        # Qui assumiamo una dimensione target standard, es. 128 o quella originale del cubo
        target_size = (img.shape[2], img.shape[3], img.shape[4]) # Usa size corrente se non downsampled
        # Se img è piccola, interpolate la ingrandisce
        # tmp_img = F.interpolate(img, size=target_size, mode="trilinear")
        tmp_img = img # Per ora lasciamo invariato
    else:
        tmp_img = img

    # 2. Slicing intelligente (Astro vs Medical)
    # perc_dim = "spatial" (piano cielo) o "spectral" (PV diagram)
    
    # [B, C, D(Freq), H(Dec), W(RA)]
    
    if hparams.slicing_dim == "spatial": 
        # Prende fetta centrale lungo l'asse spettrale (D) -> Immagine 2D (H, W)
        # Permute: [B, C, D, H, W] -> [B, D, C, H, W] -> Prendi slice centrale di D
        mid_idx = tmp_img.shape[2] // 2
        slice_2d = tmp_img[:, :, mid_idx, :, :] # [B, C, H, W]
        
    elif hparams.slicing_dim == "spectral_ra":
        # Prende fetta centrale lungo Dec (H) -> PV Diagram (D, W)
        mid_idx = tmp_img.shape[3] // 2
        slice_2d = tmp_img[:, :, :, mid_idx, :] # [B, C, D, W]
        
    else: # Default o spectral_dec
        mid_idx = tmp_img.shape[4] // 2
        slice_2d = tmp_img[:, :, :, :, mid_idx] # [B, C, D, H]

    # 3. Adattamento a VGG (3 Canali)
    # slice_2d è [B, 1, H, W]. VGG vuole [B, 3, H, W]
    slice_rgb = slice_2d.repeat(1, 3, 1, 1)

    # Features extraction
    features = vgg16(slice_rgb, resize_images=False, return_lpips=True)
    return features