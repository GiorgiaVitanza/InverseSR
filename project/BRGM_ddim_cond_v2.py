# Code adapted for Astrophysical Data Restoration
# Original Reference: Pinaya et al. (2022) & Marinescu et al. (2020)

import pandas as pd
import csv
from argparse import ArgumentParser, Namespace
import os
from pathlib import Path
from skimage.transform import resize

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import normalized_root_mse as nmse
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from models.BRGM.forward_models import ForwardDownsample
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
from utils.plot_new import draw_corrupted_images, draw_images, denormalize_data, compare_cubes, comparison_plots_ok, plot_orthogonal_cuts, draw_img
from utils.const import (
    INPUT_FOLDER_CAT
)
from visualizzazione_3d import volume_rendering


def denormalize_cond(cond: torch.Tensor, catalogue: pd.DataFrame, feature_cols: list) -> torch.Tensor:
    """
    Denormalizza i parametri basandosi sui valori reali del catalogo.
    """
    mins = [catalogue[col].min() for col in feature_cols]
    maxs = [catalogue[col].max() for col in feature_cols]
    
    stats_min = torch.tensor(mins, device=cond.device, dtype=torch.float32)
    stats_max = torch.tensor(maxs, device=cond.device, dtype=torch.float32)
    
    current_cond_phys = cond.detach() * (stats_max - stats_min) + stats_min
    return torch.clamp(current_cond_phys, stats_min, stats_max)


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
        "metrics/nmse": metrics["nmse"],
        "inv_cond/hi_size": cond_vals[0, 0].item(),
        "inv_cond/line_flux_integral": cond_vals[0, 1].item(),
        "inv_cond/i": cond_vals[0, 2].item(),
        "inv_cond/w20": cond_vals[0, 3].item(),
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
    forward: ForwardDownsample,
    target: torch.Tensor,
    device: torch.device,
    writer: SummaryWriter,
    hparams: Namespace,
    patch_stats: dict = None,
    verbose: bool = False,
):
    # 1. SETUP INIZIALE
    cat_path = Path(INPUT_FOLDER_CAT)
    cat = {}

    with open(cat_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            obj_id = row["patch_id"]
            cat[obj_id] = {
                "hi_size": float(row["hi_size"]),
                "line_flux_integral": float(row["line_flux_integral"]),
                "i": float(row["i"]),
                "w20": float(row["w20"]),
            }
    feature_cols = ['hi_size', 'line_flux_integral', 'i', 'w20']
    
    # Inizializzazione  (4 parametri fisici) e latent_variable (3 canali spaziali)
    cond, latent_variable = setup_noise_inputs(cat, device=device, hparams=hparams)
    
    # 2. PREPARAZIONE SPATIAL MASK PER IL CONCAT CONDITIONING
    # Se il target ha forma [B, C, D, H, W], prendiamo solo il primo canale per la maschera spaziale
    spatial_mask = (target[:, :1] > 0).float() if target.shape[1] > 1 else (target > 0).float()
    
    _, _, D_lat, H_lat, W_lat = latent_variable.shape

    # Interpoliamo la spatial_mask per farla combaciare con la risoluzione del latente
    mask_latent = F.interpolate(
        spatial_mask, 
        size=(D_lat, H_lat, W_lat), 
        mode='trilinear', 
        align_corners=False
    ).detach()

    # 3. IMPOSTAZIONE PARAMETRI DA OTTIMIZZARE
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

    # 4. PREPARAZIONE TARGET CORROTTO E VGG
    target_img_corrupted = forward(target)
    vgg16, target_features = load_vgg_perceptual(hparams, target_img_corrupted, device)
    
    if target_features is not None:
        target_features = target_features.detach()

    save_path = Path(hparams.output_dir_BRGM_ddim)
    save_path.mkdir(parents=True, exist_ok=True)
    # 5. OTTIMIZZAZIONE LOOP
    for step in range(hparams.start_steps, hparams.num_steps):
        def closure():
            optimizer.zero_grad()

            # A. COSTRUZIONE DEL CONDITIONING PAYLOAD AD OGNI STEP
            cond_payload = {}
            # Concatenazione Spaziale (1 canale mask_latent -> porta i canali in UNet a 3+1 = 4)
            if hparams.cond_key in ["concat", "hybrid"]:
                cond_payload["c_concat"] = [mask_latent]

            # Cross-Attention sui parametri fisici (4 parametri ottimizzati)
            if hparams.cond_key in ["crossattn", "hybrid"]:
                context_attn = cond.unsqueeze(1) if cond.ndim == 2 else cond
                cond_payload["c_crossattn"] = [context_attn]

            # B. GENERAZIONE (Latent + Cond Payload -> Image)
            synth_img = sampling_from_ddim(
                ddim=ddim,
                decoder=decoder,
                latent_variable=latent_variable,
                cond=cond_payload,  # Passiamo il dizionario completo con c_concat e c_crossattn
                hparams=hparams,
            )

            # C. CORRUZIONE E LOSS
            synth_img_corrupted = forward(synth_img)
            # pixel_loss = (synth_img_corrupted - target_img_corrupted).abs().mean()
            # Crea un peso proporzionale alla luminosità del target
            weights = torch.abs(target_img_corrupted) + 1.0  # o target_img_corrupted ** 2
            pixel_loss = ((synth_img_corrupted - target_img_corrupted).abs() * weights).mean()
            loss = pixel_loss

            """# Imposta alpha per decidere quanto pesare i picchi rispetto al background
            alpha = 5.0  

            # Calcola i pesi dal target reale
            weights = 1.0 + alpha * torch.abs(target_img_corrupted)

            # Weighted Mean Absolute Error (L1 pesata)
            diff = (synth_img_corrupted - target_img_corrupted).abs()
            pixel_loss = (diff * weights).sum() / weights.sum()

            loss = pixel_loss"""
                        
            perc_loss = torch.tensor(0.0, device=device)
            if hparams.lambda_perc > 0 and vgg16 is not None:
                synth_features = getVggFeatures(hparams, synth_img_corrupted, vgg16)
                perc_loss = (target_features - synth_features).abs().mean()
                loss += hparams.lambda_perc * perc_loss

            # D. BACKPROPAGATION
            loss.backward()
            
            # Applichiamo la maschera se vogliamo ottimizzare solo alcuni parametri di cond 
            if hparams.update_conditioning and cond.grad is not None:
                cond.grad *= mask_cond
            
            # CALCOLO METRICHE PER LOGGING
            synth_img_np = synth_img[0, 0].detach().cpu().numpy()
            target_np = target[0, 0].detach().cpu().numpy()
            synth_phys = denormalize_data(synth_img_np, norm_mode=hparams.norm_data)
            target_phys = denormalize_data(target_np, norm_mode=hparams.norm_data)

            if hparams.norm_data != 'zscore':
                ssim_ = ssim(synth_phys, target_phys, win_size=11, data_range=1.0, gaussian_weights=True, use_sample_covariance=False)
            else:
                ssim_ = ssim(synth_phys, target_phys, win_size=11, data_range=2.0, gaussian_weights=True, use_sample_covariance=False)

            global_max = max(target_phys.max(), synth_phys.max())
            global_min = min(target_phys.min(), synth_phys.min())
            data_range = global_max - global_min

            psnr_ = psnr(target_phys, synth_phys, data_range=data_range)
            mse_ = mse(target_phys, synth_phys)
            nmse_ = nmse(target_phys, synth_phys)

            writer.add_scalar("loss", loss, global_step=step)
            writer.add_scalar("pixelwise_loss", pixel_loss, global_step=step)
            writer.add_scalar("perceptual_loss", perc_loss, global_step=step)
            writer.add_scalar("ssim", ssim_, global_step=step)
            writer.add_scalar("psnr", psnr_, global_step=step)
            writer.add_scalar("mse", mse_, global_step=step)
            writer.add_scalar("nmse", nmse_, global_step=step)

            # Log parametri fisici denormalizzati
            cond_phys = denormalize_cond(cond, catalogue=pd.DataFrame.from_dict(cat, orient='index'), feature_cols=feature_cols)
            if verbose:
                print(f"Step {step:03d} | Loss: {loss.item():.6f} | Hi Size: {cond_phys[0,0]:.4f} | Line Flux Integral: {cond_phys[0,1]:.4f} | I: {cond_phys[0,2]:.4f} | W20: {cond_phys[0,3]:.4f} | SSIM_mid: {ssim_:.4f}")

            # E. LOGGING IMMAGINI
            if step % 10 == 0:

                synth_vis = denormalize_data(synth_img[0, 0].detach().cpu().numpy(), norm_mode=hparams.norm_data, patch_stats=patch_stats)
                target_vis = denormalize_data(target[0, 0].detach().cpu().numpy(), norm_mode=hparams.norm_data, patch_stats=patch_stats)
                target_img_corrupted_vis = denormalize_data(target_img_corrupted[0, 0].detach().cpu().numpy(), norm_mode=hparams.norm_data, patch_stats=patch_stats)
                synth_img_corrupted_vis = denormalize_data(synth_img_corrupted[0, 0].detach().cpu().numpy(), norm_mode=hparams.norm_data, patch_stats=patch_stats)
            
                # --- SANITIZZAZIONE NAN / INF ---
                synth_vis = np.nan_to_num(synth_vis, nan=0.0, posinf=1.0, neginf=0.0)
                target_vis = np.nan_to_num(target_vis, nan=0.0, posinf=1.0, neginf=0.0)
                synth_img_corrupted_vis = np.nan_to_num(synth_img_corrupted_vis, nan=0.0, posinf=1.0, neginf=0.0)
                target_img_corrupted_vis = np.nan_to_num(target_img_corrupted_vis, nan=0.0, posinf=1.0, neginf=0.0)
            
                
            
                final_step_str = f"{hparams.num_steps}".zfill(4)
            
                draw_img(target_vis, title=f"final_target_{hparams.norm_data}", step=final_step_str, output_folder=save_path)
                draw_img(synth_img_corrupted_vis, title=f"final_corrupted_{hparams.norm_data}", step=final_step_str, output_folder=save_path)
            
                compare_cubes(target_vis, synth_vis, title=f"target_vs_synth_{hparams.norm_data}", save_path=save_path / "compare_target_vs_synth.png")
                compare_cubes(target_img_corrupted_vis, synth_img_corrupted_vis, title=f"corr_target_vs_corr_synth_{hparams.norm_data}", save_path=save_path / "compare_corrupted_target_vs_corrupted_synth.png")
                
                fig = comparison_plots_ok(target_vis, synth_vis)
                fig.savefig(save_path / f"comparison_ok_{hparams.norm_data}.png")
                plt.close(fig)
            
                fig_corrupted = comparison_plots_ok(target_img_corrupted_vis, synth_img_corrupted_vis)
                fig_corrupted.savefig(save_path / f"comparison_corrupted_ok_{hparams.norm_data}.png")
                plt.close(fig_corrupted)
            
                plot_orthogonal_cuts(synth_vis, title=f"orthogonal_cuts_synth_{hparams.norm_data}", save_path=save_path / "orthogonal_cuts_synth.png")
                plot_orthogonal_cuts(target_vis, title=f"orthogonal_cuts_target_{hparams.norm_data}", save_path=save_path / "orthogonal_cuts_target.png")
                plot_orthogonal_cuts(synth_img_corrupted_vis, title=f"orthogonal_cuts_corrupted_{hparams.norm_data}", save_path=save_path / "orthogonal_cuts_corrupted.png")
            
                print("Plots saved to", save_path)
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

            closure.final_metrics = {"loss": loss.item(), "ssim": ssim_, "psnr": psnr_, "mse": mse_, "nmse": nmse_}
            closure.final_cond_phys = cond_phys
            
            return loss

        optimizer.step(closure=closure)
        
        # Clamp sui parametri fisici per rimanere nell'intervallo [0, 1]
        with torch.no_grad():
            cond.clamp_(0, 1)
        latent_variable_out[step] = latent_variable.detach()[0]
        cond_out[step] = cond.detach()[0]

    add_hparams_to_tensorboard(
        hparams,
        metrics=closure.final_metrics,
        cond_vals=closure.final_cond_phys,
        writer=writer,
    )

    writer.flush()
    writer.close()
      
    os.makedirs(hparams.output_dir_BRGM_ddim, exist_ok=True)
    torch.save(
        {
            "epoch": step,
            "latent_variable": latent_variable,
            "cond": cond,
            "optimizer": optimizer.state_dict(),
        },
        f"{hparams.output_dir_BRGM_ddim}/checkpoint.pth",
    )

    return latent_variable_out, cond_out, {"loss": closure.final_metrics["loss"], "ssim": closure.final_metrics["ssim"]}


def main(hparams: Namespace) -> None:
    device = torch.device(hparams.device)
    
    # Inizializza TensorBoard
    writer = SummaryWriter(log_dir=hparams.tensor_board_logger_ddim)

    # 1. Carica il target (es. FITS 128x128x128)
    img_tensor, patch_stats = load_target_image(hparams, device=device)

    # Prepara il dato (slice centrale del volume 3D)
    img_data = img_tensor[0].cpu().numpy()
    mid_slice = img_data.shape[0] // 2
    slice_to_plot = img_data[mid_slice, :, :]
    
    plt.figure(figsize=(8, 8))
    plt.imshow(slice_to_plot, cmap='hot', origin="lower") 
    plt.colorbar(label='Intensità')
    plt.title(f"Target image nel main normalizzato {hparams.norm_data} (Slice centrale)")

    os.makedirs(hparams.output_dir_BRGM_ddim, exist_ok=True)
    output_path = Path(hparams.output_dir_BRGM_ddim) 
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_immagine_output = output_path / "target_mid_slice.png"
    plt.savefig(file_immagine_output)
    plt.close() 
    
    volume_rendering(img_data, "target", output_dir=str(output_path))

    if img_tensor.ndim == 4:  # Se è [C, D, H, W], aggiungi dimensione batch -> [1, C, D, H, W]
        img_tensor = img_tensor.unsqueeze(0)
        
    # 2. Carica i modelli pre-allenati
    diffusion, decoder = load_pre_trained_model(hparams, device=device)
    ddim = DDIMSampler(diffusion)
    
    # 3. Setup Forward Model (Degradazione)
    forward = create_corruption_function(hparams=hparams, device=device)

    # 4. Esecuzione Inversione
    final_z, final_cond, metrics = project(
        ddim, decoder, forward, img_tensor, device, writer, hparams, patch_stats, verbose=True
    )

    # 5. Salvataggio latente e condizioni ottimizzate
    save_path = hparams.output_dir_BRGM_ddim
    torch.save({"z": final_z, "cond": final_cond}, f"{save_path}/results.pth")
    print(f"Risultati salvati in {save_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Inversione Diffusion Model per Dati Astrofisici")
    add_argument(parser)
    args = parser.parse_args()
    main(args)