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
from utils.plot_new import draw_corrupted_images, draw_images # 

OUTPUT_FOLDER = "./data/outputs/BRGM_ddim_cond" 

# --- HELPER FUNCTIONS ---

def logprint(message: str, verbose: bool) -> None:
    if verbose:
        print(message)


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

def create_mask_for_backprop(hparams: Namespace, device: torch.device) -> torch.Tensor:
    mask_cond = torch.ones((1, 4), device=device)
    mask_cond[:, 0] = 0 if not hparams.update_hi_size else 1
    mask_cond[:, 1] = 0 if not hparams.update_line_flux_integral else 1
    mask_cond[:, 2] = 0 if not hparams.update_i else 1
    mask_cond[:, 3] = 0 if not hparams.update_w20 else 1
    return mask_cond

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
    if hparams.update_latent_variables:
        latent_variable.requires_grad = True
        update_params.append(latent_variable)
    if hparams.update_conditioning:
        cond.requires_grad = True
        update_params.append(cond)

    

    optimizer = torch.optim.Adam(
        update_params, 
        betas=(0.9, 0.999), 
        lr=hparams.learning_rate
    )

    latent_variable_out = torch.zeros(
        [hparams.num_steps] + list(latent_variable.shape[1:]),
        dtype=torch.float32,
        device=device,
    )
    cond_out = torch.zeros(
        [hparams.num_steps] + list(cond.shape[1:]),
        dtype=torch.float32,
        device=device,
    )

    mask_cond = create_mask_for_backprop(hparams, device)

    # 2. PREPARAZIONE TARGET
    target_img_corrupted = forward(target)
    vgg16, target_features = load_vgg_perceptual(hparams, target_img_corrupted, device)
    
    if target_features is not None:
        target_features = target_features.detach()

    # 3. OTTIMIZZAZIONE LOOP
    for step in range(hparams.start_steps, hparams.num_steps):
        def closure():
            optimizer.zero_grad()


            # B. GENERAZIONE (Latent -> Image)
            synth_img = sampling_from_ddim(
                ddim=ddim,
                decoder=decoder,
                latent_variable=latent_variable,
                cond=cond, 
                hparams=hparams,
            )

            # C. CORRUZIONE E LOSS
            synth_img_corrupted = forward(synth_img)
            loss = 0
            prior_loss = 0
            

            pixel_loss = (synth_img_corrupted - target_img_corrupted).abs().mean()
            loss = pixel_loss
            
            if hparams.lambda_perc > 0 and vgg16 is not None:
                synth_features = getVggFeatures(hparams, synth_img_corrupted, vgg16)
                perc_loss = (target_features - synth_features).abs().mean()
                loss += hparams.lambda_perc * perc_loss

            # D. BACKPROPAGATION
            loss.backward()
            
            # Applichiamo la maschera se vogliamo ottimizzare solo alcuni parametri di cond 
            if hparams.update_conditioning and cond.grad is not None:
                cond.grad *= mask_cond
            return (
                    loss,
                    pixel_loss,
                    perc_loss,
                    prior_loss,
                    synth_img,
                    synth_img_corrupted,
            )    
        (
            loss,
            pixel_loss,
            perc_loss,
            prior_loss,
            synth_img,
            synth_img_corrupted,
        ) = optimizer.step(closure=closure)

        synth_img_np = synth_img[0, 0].detach().cpu().numpy()
        target_np = target[0, 0].detach().cpu().numpy()
        ssim_ = ssim(
            synth_img_np,
            target_np,
            win_size=11,
            data_range=1.0,
            gaussian_weights=True,
            use_sample_covariance=False,
        )
        # Code for computing PSNR is adapted from
        # https://github.com/agis85/multimodal_brain_synthesis/blob/master/error_metrics.py#L32
        data_range = np.max([synth_img_np.max(), target_np.max()]) - np.min(
            [synth_img_np.min(), target_np.min()]
        )
        psnr_ = psnr(target_np, synth_img_np, data_range=data_range)
        mse_ = mse(target_np, synth_img_np)
        nmse_ = nmse(target_np, synth_img_np)

        writer.add_scalar("loss", loss, global_step=step)
        writer.add_scalar("pixelwise_loss", pixel_loss, global_step=step)
        writer.add_scalar("perceptual_loss", perc_loss, global_step=step)
        writer.add_scalar("prior_loss", prior_loss, global_step=step)
        writer.add_scalar("ssim", ssim_, global_step=step)
        writer.add_scalar("psnr", psnr_, global_step=step)
        writer.add_scalar("mse", mse_, global_step=step)
        writer.add_scalar("nmse", nmse_, global_step=step)

        # E. LOGGING
        if step % 10 == 0:
            if hparams.corruption != "None":
                imgs = draw_corrupted_images(
                    synth_img_np,
                    target_np,
                    synth_img_corrupted[0, 0].detach().cpu().numpy(),
                    target_img_corrupted[0, 0].detach().cpu().numpy(),
                    ssim_=ssim_,
                )
            else:
                imgs = draw_images(
                    synth_img_np,
                    target_np,
                    ssim_=ssim_,
                )
            step_ = f"{step}".zfill(4)
            writer.add_figure(f"step: {step_}", imgs, global_step=step)
            plt.close(imgs)

        latent_variable_out[step] = latent_variable.detach()[0]
        cond_out[step] = cond.detach()[0]
        writer.add_scalar("loss/total", loss.item(), step)
        writer.add_scalar("metrics/ssim_mid", ssim_, step)
        # Logghiamo i parametri fisici correnti 
        writer.add_scalar("inv_cond/hi_size", cond[0, 0].item(), step)
        writer.add_scalar("inv_cond/line_flux_integral", cond[0, 1].item(), step)
        writer.add_scalar("inv_cond/i", cond[0, 2].item(), step)
        writer.add_scalar("inv_cond/w20", cond[0, 3].item(), step)

        writer.flush()
        writer.close()
        if verbose:
                    print(f"Step {step:03d} | Loss: {loss.item():.6f} | Hi Size: {cond[0,0]:.4f} | Line Flux Integral: {cond[0,1]:.4f} | I: {cond[0,2]:.4f} | W20: {cond[0,3]:.4f} | SSIM_mid: {ssim_:.4f}")


    torch.save(
        {
            "epoch": step,
            "latent_variable": latent_variable,
            "cond": cond,
            "optimizer": optimizer.state_dict(),
        },
        OUTPUT_FOLDER / "checkpoint.pth",
    )

    return latent_variable_out, cond_out, {"loss": loss.item(), "ssim": ssim_}


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
