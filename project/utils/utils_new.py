import os
import random
from argparse import Namespace
from pathlib import Path
from typing import Tuple, Dict, List, Any

import mlflow
import numpy as np
import torch
from monai.transforms import apply_transform
import pandas as pd


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
    INPUT_FOLDER_TEST,
    MASK_FOLDER,
    PRETRAINED_MODEL_VGG_PATH,
)
from utils.dataset_v3 import normalize_dynamic


def generating_latent_vector(
    diffusion: torch.nn.Module,
    latent_variable: torch.Tensor,
    conditioning: Dict[str, List[torch.Tensor]],
    batch_size: int,
    image_size: Tuple[int, int, int],
    scale_factor: int,
    z_channels: int,
):
    ddim = DDIMSampler(diffusion)
    num_timesteps = 50
    latent_vectors, _ = ddim.sample(
        S=num_timesteps,
        batch_size=batch_size,        
        shape=[z_channels, image_size[0] // scale_factor, image_size[1] // scale_factor, image_size[2] // scale_factor],
        first_img=latent_variable,
        conditioning=conditioning,        
        eta=1.0,
        verbose=False,
    )

    return latent_vectors

def inference(
    vqvae: Any,
    latent_vectors: torch.Tensor,
):
    x_hat = vqvae.reconstruct_ldm_outputs(latent_vectors)
    return x_hat

def load_ddpm_latent_vectors(device: torch.device, hparams: Namespace) -> torch.Tensor:
    checkpoint = torch.load(
        Path(hparams.path_to_latent_ddpm),
        map_location=device,
        weights_only=False,
    )
    
    # Estraiamo il tensore latente usando la chiave "z"
    if "z" in checkpoint:
        ddpm_latent_vectors = checkpoint["z"]
        print(f"Latente caricato con successo. Shape: {ddpm_latent_vectors.shape}")
    else:
        raise KeyError(f"Errore: chiave 'z' non trovata nel file. Chiavi presenti: {list(checkpoint.keys())}")
        
    return ddpm_latent_vectors

def load_ddpm_model(ddpm_path: Path, device: torch.device) -> torch.nn.Module:
    # 1. Caricamento del modello tramite MLflow
    diffusion = mlflow.pytorch.load_model(
        str(ddpm_path),
        map_location=device,
    )
    
    # 2. ATTIVAZIONE GRADIENT CHECKPOINTING (Cruciale per i layer di Attention)
    # Cerchiamo di attivarlo nella UNet interna del modello DDPM
    if hasattr(diffusion, 'model') and hasattr(diffusion.model, 'diffusion_model'):
        # Questo è il percorso tipico per implementazioni stile LDM/Stable Diffusion
        diffusion.model.diffusion_model.use_checkpoint = True
    elif hasattr(diffusion, 'use_checkpoint'):
        diffusion.use_checkpoint = True

    # 3. Preparazione modello
    diffusion.eval()
    diffusion = diffusion.to(device)
    diffusion.requires_grad_(False)
    
    # 4. Ottimizzazione della memoria (Opzionale ma consigliato)
    # Converte i pesi in Half Precision (float16) per risparmiare il 50% di VRAM
    # diffusion = diffusion.half() 
    
    return diffusion

def load_pre_trained_decoder(
    vae_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    vqvae = mlflow.pytorch.load_model(
        str(vae_path),
        map_location=device,
    )
    vqvae.eval()
    vqvae = vqvae.to(device)
    vqvae.requires_grad_(False)
    return vqvae
#-------------------------------------------------------------------

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



def load_target_image(hparams: Namespace, device: torch.device) -> torch.Tensor:
    """
    Carica l'immagine target e applica la normalizzazione specificata in hparams.norm_mode.
    """
    # 1. IDENTIFICAZIONE FILE (Mantengo la tua logica esistente)
    if hparams.data_format == "npy" and hparams.inference:
        potential_files = list(INPUT_FOLDER_PATCHES.glob(f"*.npy"))
        print("Inference mode: loading npy patches")
    elif hparams.data_format == "npy" and hparams.test_mode:
        potential_files = list(INPUT_FOLDER_TEST.glob(f"*.npy"))
    elif hparams.data_format == "fits":
        potential_files = list(INPUT_FOLDER_PATCHES.glob(f"*{hparams.object_id}*.fits"))
    else:
        raise ValueError(f"Formato {hparams.data_format} non supportato.")

    if not potential_files:
        raise FileNotFoundError(f"Nessun file trovato per {hparams.object_id}")

    img_path = potential_files[0]
    print(f"Caricamento immagine target da: {img_path}")
    
    # 2. CARICAMENTO DATI RAW
    if hparams.data_format == "fits":
        # Assumendo che transform_img carichi il FITS e lo porti su device
        img_tensor = transform_img(img_path, device=device)
    else:
        data = np.load(img_path).astype(np.float32)
        img_tensor = torch.from_numpy(data).to(device)


    # 3. APPLICAZIONE NORMALIZZAZIONE MULTI-MODE
    norm_mode = hparams.norm_data 
    img_tensor[2:5] = normalize_dynamic(img_tensor[2:5], norm_mode=norm_mode)

    if norm_mode != 'zscore':
        print('carico il target nel range [0, 1]')
        img_tensor[2:5] = torch.clamp(img_tensor[2:5], 0, 1)
    else:
        print('carico il target nel range [-1,1]')
        img_tensor[2:5] = torch.clamp(img_tensor[2:5], -1, 1)
        
    return img_tensor
        
    


def load_pre_trained_model(hparams: Namespace, device: torch.device) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """Carica VAE (Decoder) e DDPM tramite MLFlow."""
    print(f"Caricamento modelli da:\nVAE: {hparams.vae_path_BRGM}\nDDPM: {hparams.ddpm_path_BRGM}")
    
    decoder = mlflow.pytorch.load_model(str(hparams.vae_path_BRGM), map_location=device)
    ddpm = mlflow.pytorch.load_model(str(hparams.ddpm_path_BRGM), map_location=device)
    
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

def setup_noise_inputs(cat, device: torch.device, hparams: Namespace) -> Tuple[torch.Tensor, torch.Tensor]:
    # 1. Valori grezzi (Raw) dal catalogo
    # Nota: Assicurati che 'patch_000000.npy' sia dinamico o passato correttamente
    obj_data = cat[hparams.object_id] 
    cond_list = [
        obj_data['hi_size'], 
        obj_data['line_flux_integral'], 
        obj_data['i'], 
        obj_data['w20']
    ]
    cond_raw = torch.tensor([cond_list], device=device, dtype=torch.float32)
    
    # 2. Calcolo DINAMICO di mins e maxs dal catalogo
   
    df_cat = pd.DataFrame.from_dict(cat, orient='index')
    feature_cols = ['hi_size', 'line_flux_integral', 'i', 'w20']
    
    # Calcoliamo i valori reali presenti nel file corrente
    mins = torch.tensor(df_cat[feature_cols].min().values, device=device, dtype=torch.float32)
    maxs = torch.tensor(df_cat[feature_cols].max().values, device=device, dtype=torch.float32)

    # 3. Normalizzazione Min-Max (0-1)
    # Formula: (x - min) / (max - min)
    cond_normalized = (cond_raw - mins) / (maxs - mins + 1e-8)

    # 4. Abilitiamo il gradiente
    # Tip: Clamping durante l'ottimizzazione aiuterà a non uscire dal range [0, 1]
    cond_normalized.requires_grad_(True)

    # --- Gestione Latente ---
    f = hparams.downsample_factor if hparams.corruption == "downsample" else 1
    latent_shape = (1, hparams.z_channels, hparams.image_size[0]//f, hparams.image_size[1]//f, hparams.image_size[2]//f)
    latent_variable = torch.randn(latent_shape, device=device, requires_grad=True)
    
    return cond_normalized, latent_variable

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
    dim_1 = hparams.image_size[0] // hparams.downsample_factor if hparams.corruption == "downsample" else hparams.image_size[0]
    dim_2 = hparams.image_size[1] // hparams.downsample_factor if hparams.corruption == "downsample" else hparams.image_size[1]
    dim_3 = hparams.image_size[2] // hparams.downsample_factor if hparams.corruption == "downsample" else hparams.image_size[2]
    cond_concat = cond_concat.expand(-1, -1, dim_1, dim_2, dim_3) # [1, 4, 32, 32, 32]

    conditioning = {
        "c_concat": [cond_concat],
        "c_crossattn": [cond_crossatten],
    }
    
    latent_vectors, _ = ddim.sample(
        S=hparams.ddim_num_timesteps,
        conditioning=conditioning,
        batch_size=1,
        shape=[hparams.z_channels, dim_1, dim_2, dim_3], # Esclude dimensione Batch
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
    if hparams.out_channels == 3:
        from utils.vgg_gen_3ch import AstroVGG_Slim
        vgg16 = AstroVGG_Slim("././data/trained_models_astro/vgg/vgg16_slim_astro.pth",in_channels=3, num_blocks=2).to(device)
    elif hparams.out_channels == 1:
        from utils.vgg_gen_1ch import AstroVGG_Slim
        vgg16 = AstroVGG_Slim("././data/trained_models_astro/vgg/vgg16_slim_astro_1ch.pth",in_channels=1, num_blocks=2).to(device)
    #vgg16 = AstroVGG_Slim("././data/trained_models_astro/vgg/vgg16_slim_astro.pth",in_channels=3, num_blocks=2).to(device)
    
    # 2. Carica i pesi 
    vgg16.load_state_dict(torch.load(PRETRAINED_MODEL_VGG_PATH, map_location=device, weights_only=False))
    vgg16.eval()

    # Calcola le feature del target
    target_features = getVggFeatures(hparams, target, vgg16)
    return vgg16, target_features

def getVggFeatures(hparams, img, vgg16):
    
    mid_idx = img.shape[2] // 2 
    slice_2d = img[:, 0:1, mid_idx, :, :] 
    
    features = vgg16(slice_2d)
    return features
