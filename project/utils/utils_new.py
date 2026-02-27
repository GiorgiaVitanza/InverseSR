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
from utils.vgg_gen_new import AstroVGG_Slim
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

    # --- NUOVA LOGICA PER I 3 CANALI ---
    # Se img_tensor è (B, 1, D, H, W), lo portiamo a (B, 3, D, H, W)
    if img_tensor.shape[1] == 1:
        # Duplichiamo il canale singolo 3 volte lungo la dimensione C
        img_tensor = img_tensor.repeat(1, 3, 1, 1, 1)

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
    # Creiamo un vettore di 4 parametri (es. tutti a 0.5 o valori medi del catalogo)
    cond_full = torch.full((1, 4), 0.5, device=device, dtype=torch.float32)
    
    # Se vuoi ottimizzare i primi due (es. Freq e Flux sono i primi due nel tuo dataset)
    # li rendiamo parte del grafo di calcolo
    cond_full.requires_grad_(True)

    # Rumore latente (z_channels=3, resolution=32)
    
    f = hparams.downsample_factor if hparams.corruption == "downsample" else 1
    # Se hparams.image_size è [160, 224, 160]
    latent_depth = hparams.image_size[0] // f  # 20
    latent_height = hparams.image_size[1] // f # 28
    latent_width = hparams.image_size[2] // f  # 20

    latent_variable = torch.randn(
        (1, hparams.z_channels, latent_depth, latent_height, latent_width), 
        device=device, requires_grad=True
    )
    #latent_variable = torch.randn(latent_shape, device=device, requires_grad=True)
    
    return cond_full, latent_variable


def sampling_from_ddim(
    ddim: DDIMSampler,
    latent_variable: torch.Tensor,
    decoder: OnlyDecoder,
    cond: torch.Tensor,
    hparams: Namespace,
) -> torch.Tensor:
    # 1. Cross-Attention: [Batch, Sequence, Features] -> [1, 1, 4]
    cond_crossatten = cond.unsqueeze(1) 
    
    # 2. Concatenazione spaziale: [1, 4, 1, 1, 1]
    cond_concat = cond.view(1, 4, 1, 1, 1)
    cond_concat = cond_concat.expand(-1, -1, 40, 56, 40) # [1, 4, 32, 32, 32]

    conditioning = {
        "c_concat": [cond_concat],
        "c_crossattn": [cond_crossatten],
    }
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
        
     
    
    if hasattr(latent_vectors, "as_tensor"):
        latent_vectors = latent_vectors.as_tensor()

    # Passaggio a FP16 per il decoding (risparmia VRAM)
    decoder.float()
    latent_vectors = latent_vectors.float()

    astro_img = decoder.reconstruct_ldm_outputs(latent_vectors)
     
    return astro_img

# --- PERCEPTUAL LOSS UTILS (VGG) ---

def load_vgg_perceptual(hparams: Namespace, target: torch.Tensor, device: torch.device) -> Tuple[Any, torch.Tensor]:
    """Carica la versione Slim di VGG16 per dati Astro."""
    
    # 1. Istanzia il modello (usa lo stesso numero di blocchi dello script di generazione)
    vgg16 = AstroVGG_Slim("././data/trained_models_astro/vgg/vgg16_slim_astro.pth",in_channels=3, num_blocks=2).to(device)
    
    # 2. Carica i pesi (state_dict invece di JIT)
    # Assicurati che PRETRAINED_MODEL_VGG_PATH punti al file .pth generato prima
    vgg16.load_state_dict(torch.load(PRETRAINED_MODEL_VGG_PATH, map_location=device))
    vgg16.eval()

    # Calcola le feature del target
    target_features = getVggFeatures(hparams, target, vgg16)
    return vgg16, target_features

def getVggFeatures(hparams, img, vgg16):
    # img è [1, 3, D, H, W]
    
    # 1. Slicing (Prendiamo la fetta centrale della profondità)
    mid_idx = img.shape[2] // 2 
    slice_2d = img[:, :, mid_idx, :, :] # Risultato: [1, 3, H, W]
    
    # 2. Passaggio alla VGG (Ora i canali corrispondono: 3 == 3)
    features = vgg16(slice_2d)
    return features
