# Code adapted for Astrophysical Data Restoration
# Original Reference: Pinaya et al. (2022) & Marinescu et al. (2020)

import math
import csv
import gc
from argparse import ArgumentParser, Namespace
from pathlib import Path
from time import perf_counter
from typing import Any, Tuple
from skimage.transform import resize

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import normalized_root_mse as nmse
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# --- CUSTOM MODULES (Assicurati che i file utils siano aggiornati come discusso) ---
from models.BRGM.forward_models import (
    ForwardDownsample,
    ForwardFillMask,
    ForwardAbstract,
)
from models.ddim import DDIMSampler
from utils.add_argument import add_argument
from utils.utils_new import (
    setup_noise_inputs,
    load_target_image,
    load_pre_trained_model,
    create_corruption_function,
    sampling_from_ddim,
    getVggFeatures,
    load_vgg_perceptual
)
from utils.plot_new import compare_cubes, plot_orthogonal_cuts # <--- Nuove funzioni plot

OUTPUT_FOLDER = "./data/outputs" # Assicurati che questa cartella esista o venga creata

# --- HELPER FUNCTIONS ---

def logprint(message: str, verbose: bool) -> None:
    if verbose:
        print(message)


def create_mask_for_backprop(hparams: Namespace, device: torch.device) -> torch.Tensor:
    mask_cond = torch.ones((1, 4), device=device)
    mask_cond[:, 0] = 0 if not hparams.update_hi_size else 1
    mask_cond[:, 1] = 0 if not hparams.update_line_flux_integral else 1
    mask_cond[:, 2] = 0 if not hparams.update_i else 1
    mask_cond[:, 3] = 0 if not hparams.update_w20 else 1
    return mask_cond

def add_hparams_to_tensorboard(
    hparams: Namespace,
    metrics: dict,
    cond_vals: torch.Tensor,
    writer: SummaryWriter,
) -> None:
    """Logga i parametri e le metriche finali su TensorBoard."""
    
    hparam_dict = {
        "lr": hparams.learning_rate,
        "obj_id": hparams.object_id,
        "lambda_perc": hparams.lambda_perc,
        "steps": hparams.num_steps,
    }
    
    metric_dict = {
        "loss/final": metrics["loss"],
        "metrics/ssim": metrics["ssim"],
        "metrics/psnr": metrics["psnr"],
        "metrics/mse": metrics["mse"],
        "inv_cond/hi_size": cond_vals[0].item(),
        "inv_cond/line_flux_integral": cond_vals[1].item(),
        "inv_cond/i": cond_vals[2].item(),
        "inv_cond/w20": cond_vals[3].item(),
    }
    
    writer.add_hparams(hparam_dict, metric_dict)


def project(
    ddim: DDIMSampler,
    decoder: torch.nn.Module,
    forward: ForwardAbstract,
    target: torch.Tensor,
    device: torch.device,
    writer: SummaryWriter,
    hparams: Namespace,
    verbose: bool = False,
):
    # 1. SETUP INIZIALE
    # setup_noise_inputs ora restituisce cond [1, 4] e latent [1, 3, ...]
    cond, latent_variable = setup_noise_inputs(device=device, hparams=hparams)

    update_params = []
    up_latent = getattr(hparams, 'update_latent_variables', True)
    up_cond = getattr(hparams, 'update_conditioning', True)

    if up_latent:
        latent_variable.requires_grad_(True)
        update_params.append(latent_variable)
    
    if up_cond:
        cond.requires_grad_(True)
        update_params.append(cond)

    

    optimizer = torch.optim.Adam(update_params, betas=(0.9, 0.999), lr=hparams.learning_rate)
    mask_cond = create_mask_for_backprop(hparams, device)

    # 2. PREPARAZIONE TARGET
    target_img_corrupted = forward(target)
    vgg16, target_features = load_vgg_perceptual(hparams, target_img_corrupted, device)
    if target_features is not None:
        target_features = target_features.detach()

    # 3. OTTIMIZZAZIONE LOOP
    for step in range(hparams.num_steps):
        optimizer.zero_grad()


        # B. GENERAZIONE (Latent -> Image)
        synth_img = sampling_from_ddim(
            ddim=ddim,
            decoder=decoder,
            latent_variable=latent_variable,
            cond=cond, # <--- Usiamo cond
            hparams=hparams,
        )

        # C. CORRUZIONE E LOSS
        synth_img_corrupted = forward(synth_img)
        
        pixel_loss = (synth_img_corrupted - target_img_corrupted).abs().mean()
        

        loss = pixel_loss
        
        if hparams.lambda_perc > 0 and vgg16 is not None:
            synth_features = getVggFeatures(hparams, synth_img_corrupted, vgg16)
            perc_loss = (target_features - synth_features).abs().mean()
            loss += hparams.lambda_perc * perc_loss

        # D. BACKPROPAGATION
        loss.backward()
        
        # Applichiamo la maschera se vogliamo ottimizzare solo alcuni parametri di cond 
        if up_cond and cond.grad is not None:
             cond.grad *= mask_cond
             
        optimizer.step()

        # E. LOGGING
        if step % 10 == 0:
            with torch.no_grad():
                synth_np = synth_img[0, 0].cpu().numpy()
                target_np = target[0, 0].cpu().numpy()
                # --- All'interno del loop di project, sezione metriche ---
                # FIX: Se le dimensioni non coincidono, ridimensioniamo synth per SSIM
                if target_np.shape != synth_np.shape:
                    # Ridimensioniamo la sintesi per matchare il target
                    synth_np = resize(synth_np, target_np.shape, anti_aliasing=True)

                    # Ora il calcolo SSIM non fallirà più
                    mid = target_np.shape[0] // 2
                    ssim_val = ssim(target_np[mid], synth_np[mid], data_range=target_np.max() - target_np.min())
                else:
                    mid = synth_np.shape[0] // 2
                    drange = max(target_np.max() - target_np.min(), 1e-5)
                    ssim_val = ssim(target_np[mid], synth_np[mid], data_range=drange)
                
                writer.add_scalar("loss/total", loss.item(), step)
                writer.add_scalar("metrics/ssim_mid", ssim_val, step)
                # Logghiamo i parametri fisici correnti 
                writer.add_scalar("inv_cond/hi_size", cond[0, 0].item(), step)
                writer.add_scalar("inv_cond/line_flux_integral", cond[0, 1].item(), step)
                writer.add_scalar("inv_cond/i", cond[0, 2].item(), step)
                writer.add_scalar("inv_cond/w20", cond[0, 3].item(), step)

                if verbose:
                    print(f"Step {step:03d} | Loss: {loss.item():.6f} | Hi Size: {cond[0,0]:.4f} | Line Flux Integral: {cond[0,1]:.4f} | I: {cond[0,2]:.4f} | W20: {cond[0,3]:.4f} | SSIM_mid: {ssim_val:.4f}")

    return latent_variable, cond, {"loss": loss.item(), "ssim": ssim_val}


def main(hparams: Namespace) -> None:
    device = torch.device(hparams.device)
    
    # Inizializza TensorBoard
    writer = SummaryWriter(log_dir=hparams.tensor_board_logger)

    # 1. Carica il target (es. FITS 128x128x128)
    img_tensor = load_target_image(hparams, device=device)

    # 2. Carica i modelli pre-allenati con la tua architettura (z_channels=3)
    # Questa funzione deve inizializzare il VAE con n_channels=32, z_channels=3, etc.
    diffusion, decoder = load_pre_trained_model(device=device)
    ddim = DDIMSampler(diffusion)
    
    # 3. Setup Forward Model (Degradazione)
    forward = create_corruption_function(hparams=hparams, device=device)

    # 4. Esecuzione Inversione
    final_z, final_cond, metrics = project(
        ddim, decoder, forward, img_tensor, device, writer, hparams, verbose=True
    )

    # 5. Salvataggio latente e condizioni ottimizzate
    save_path = Path(OUTPUT_FOLDER) / hparams.experiment_name
    save_path.mkdir(parents=True, exist_ok=True)
    torch.save({"z": final_z, "cond": final_cond}, save_path / "results.pth")
    print(f"Risultati salvati in {save_path}")

if __name__ == "__main__":
    parser = ArgumentParser(description="Inversione Diffusion Model per Dati Astrofisici")
    add_argument(parser) # Assicurati che questa funzione aggiunga tutti gli argomenti necessari
    args = parser.parse_args()
    main(args)
