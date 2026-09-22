# Code adapted for Astrophysical Data Restoration
# Original Reference: Pinaya et al. (2022) & Marinescu et al. (2020)

import argparse
import csv
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import normalized_root_mse as nmse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from models.BRGM.forward_models import ForwardDownsample
from models.ddim import DDIMSampler
from utils.add_argument import add_argument
from utils.const import INPUT_FOLDER_CAT
from utils.plot_new import (
    compare_cubes,
    comparison_plots_ok,
    denormalize_data,
    draw_corrupted_images,
    draw_images,
    draw_img,
    plot_orthogonal_cuts,
)
from utils.utils_new import (
    create_corruption_function,
    getVggFeatures,
    load_pre_trained_model,
    load_target_image,
    load_vgg_perceptual,
    sampling_from_ddim,
    setup_noise_inputs,
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
    # 1. SETUP INIZIALE CATALOGO
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
    cat_df = pd.DataFrame.from_dict(cat, orient='index')
    
    # Inizializzazione cond e latent_variable
    cond, latent_variable = setup_noise_inputs(cat, device=device, hparams=hparams)
    
    # 2. PREPARAZIONE SPATIAL MASK PER IL CONCAT CONDITIONING
    spatial_mask = (target[:, :1] > 0).float() if target.shape[1] > 1 else (target > 0).float()
    _, _, D_lat, H_lat, W_lat = latent_variable.shape

    mask_latent = F.interpolate(
        spatial_mask, 
        size=(D_lat, H_lat, W_lat), 
        mode='trilinear', 
        align_corners=False
    ).detach()

    # 3. IMPOSTAZIONE PARAMETRI DA OTTIMIZZARE E OTTIMIZZATORE
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

    # --- INIZIALIZZAZIONE SCHEDULER ---
    # Decadimento Cosine Annealing che porta il LR dal valore iniziale fino a 1e-6
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=hparams.num_steps - hparams.start_steps, 
        eta_min=1e-6
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

    latest_metrics = {}
    latest_cond_phys = None
    final_synth_img = None

    # Pre-calcola target_phys per risparmiare tempo nelle metriche
    target_np = target[0, 0].detach().cpu().numpy()
    target_phys = denormalize_data(target_np, norm_mode=hparams.norm_data, patch_stats=patch_stats)

    # 5. OTTIMIZZAZIONE LOOP
    for step in range(hparams.start_steps, hparams.num_steps):
        
        # Variabili di supporto per estrarre i tensor dalla closure
        step_synth_img = None
        step_synth_img_corrupted = None
        step_pixel_loss = None
        step_perc_loss = None
        
        def closure():
            nonlocal step_synth_img, step_synth_img_corrupted, step_pixel_loss, step_perc_loss
            optimizer.zero_grad()

            # A. CONDITIONING PAYLOAD
            cond_payload = {}
            if hparams.cond_key in ["concat", "hybrid"]:
                cond_payload["c_concat"] = [mask_latent]

            if hparams.cond_key in ["crossattn", "hybrid"]:
                context_attn = cond.unsqueeze(1) if cond.ndim == 2 else cond
                cond_payload["c_crossattn"] = [context_attn]

            # B. GENERAZIONE
            synth_img = sampling_from_ddim(
                ddim=ddim,
                decoder=decoder,
                latent_variable=latent_variable,
                cond=cond_payload,
                hparams=hparams,
            )

            # C. CORRUZIONE E LOSS PESATA
            synth_img_corrupted = forward(synth_img)
            weights = torch.abs(target_img_corrupted) + 1.0
            pixel_loss = ((synth_img_corrupted - target_img_corrupted).abs() * weights).mean()
            loss = pixel_loss

            perc_loss = torch.tensor(0.0, device=device)
            if hparams.lambda_perc > 0 and vgg16 is not None:
                synth_features = getVggFeatures(hparams, synth_img_corrupted, vgg16)
                perc_loss = (target_features - synth_features).abs().mean()
                loss += hparams.lambda_perc * perc_loss

            # D. BACKPROPAGATION
            loss.backward()
            
            if hparams.update_conditioning and cond.grad is not None:
                cond.grad *= mask_cond

            # Salva riferimenti per l'analisi fuori dalla closure
            step_synth_img = synth_img
            step_synth_img_corrupted = synth_img_corrupted
            step_pixel_loss = pixel_loss.item()
            step_perc_loss = perc_loss.item()
            
            return loss

        # Esecuzione passo di ottimizzazione
        loss_tensor = optimizer.step(closure=closure)
        current_loss = loss_tensor.item()

        # --- STEP DELLO SCHEDULER ---
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        # Salva l'ultimo synth generato
        final_synth_img = step_synth_img

        # Clamp delle variabili fisiche condizionate
        with torch.no_grad():
            cond.clamp_(0, 1)
        latent_variable_out[step] = latent_variable.detach()[0]
        cond_out[step] = cond.detach()[0]

        # F. CALCOLO METRICHE & UNICO LOGGING SU TENSORBOARD AD OGNI STEP
        with torch.no_grad():
            synth_img_np = step_synth_img[0, 0].detach().cpu().numpy()
            synth_phys = denormalize_data(synth_img_np, norm_mode=hparams.norm_data, patch_stats=patch_stats)

            ssim_range = 1.0 if hparams.norm_data != 'zscore' else 2.0
            ssim_ = ssim(synth_phys, target_phys, win_size=11, data_range=ssim_range, gaussian_weights=True, use_sample_covariance=False)

            global_max = max(target_phys.max(), synth_phys.max())
            global_min = min(target_phys.min(), synth_phys.min())
            data_range = global_max - global_min if global_max > global_min else 1e-5

            psnr_ = psnr(target_phys, synth_phys, data_range=data_range)
            mse_ = mse(target_phys, synth_phys)
            nmse_ = nmse(target_phys, synth_phys)

            cond_phys = denormalize_cond(cond, catalogue=cat_df, feature_cols=feature_cols)

            # Unica scrittura scalari per ogni singola epoca/step su TensorBoard
            writer.add_scalar("Loss/Total", current_loss, global_step=step)
            writer.add_scalar("Loss/Pixelwise", step_pixel_loss, global_step=step)
            writer.add_scalar("Loss/Perceptual", step_perc_loss, global_step=step)
            writer.add_scalar("Params/LearningRate", current_lr, global_step=step)
            writer.add_scalar("Metrics/SSIM", ssim_, global_step=step)
            writer.add_scalar("Metrics/PSNR", psnr_, global_step=step)
            writer.add_scalar("Metrics/MSE", mse_, global_step=step)
            writer.add_scalar("Metrics/NMSE", nmse_, global_step=step)

            if verbose:
                print(f"Step {step:03d} | LR: {current_lr:.2e} | Loss: {current_loss:.6f} | Hi Size: {cond_phys[0,0]:.4f} | Line Flux: {cond_phys[0,1]:.4f} | I: {cond_phys[0,2]:.4f} | W20: {cond_phys[0,3]:.4f} | SSIM: {ssim_:.4f}")

            # G. ESPORTAZIONE GRAFICI PESANTI (Ogni N Step)
            if step % 50 == 0 or step == hparams.num_steps - 1:
                target_img_corrupted_vis = denormalize_data(target_img_corrupted[0, 0].detach().cpu().numpy(), norm_mode=hparams.norm_data, patch_stats=patch_stats)
                synth_img_corrupted_vis = denormalize_data(step_synth_img_corrupted[0, 0].detach().cpu().numpy(), norm_mode=hparams.norm_data, patch_stats=patch_stats)

                step_str = f"{step}".zfill(4)
                draw_img(target_phys, title=f"target_{hparams.norm_data}", step=step_str, output_folder=save_path)
                draw_img(synth_img_corrupted_vis, title=f"corrupted_{hparams.norm_data}", step=step_str, output_folder=save_path)

                compare_cubes(target_phys, synth_phys, title=f"target_vs_synth_{hparams.norm_data}", save_path=save_path / f"compare_target_vs_synth_{step_str}.png")
                
                fig = comparison_plots_ok(target_phys, synth_phys, x_lr=target_img_corrupted_vis)
                fig.savefig(save_path / f"comparison_ok_{hparams.norm_data}_{step_str}.png")
                plt.close(fig)

                plot_orthogonal_cuts(synth_phys, title=f"orthogonal_cuts_synth_{hparams.norm_data}", save_path=save_path / f"orthogonal_cuts_synth_{step_str}.png")

                if hparams.corruption != "None":
                    imgs = draw_corrupted_images(
                        synth_img_np, target_np,
                        step_synth_img_corrupted[0, 0].detach().cpu().numpy(),
                        target_img_corrupted[0, 0].detach().cpu().numpy(),
                        ssim_=ssim_,
                    )
                else:
                    imgs = draw_images(synth_img_np, target_np, ssim_=ssim_)
                
                writer.add_figure("Reconstruction", imgs, global_step=step)
                plt.close(imgs)

            latest_metrics = {"loss": current_loss, "ssim": ssim_, "psnr": psnr_, "mse": mse_, "nmse": nmse_}
            latest_cond_phys = cond_phys

    writer.flush()
    writer.close()

    # =========================================================================
    # SALVATAGGIO DATI RIPRISTINATI / CORROTTI / INPUT IN UNA CARTELLA DEDICATA
    # =========================================================================
    results_dir = save_path / "restoration_outputs"
    results_dir.mkdir(parents=True, exist_ok=True)

    target_corrupted_np = target_img_corrupted[0, 0].detach().cpu().numpy()
    target_corrupted_phys = denormalize_data(target_corrupted_np, norm_mode=hparams.norm_data, patch_stats=patch_stats)

    final_synth_np = final_synth_img[0, 0].detach().cpu().numpy()
    final_synth_phys = denormalize_data(final_synth_np, norm_mode=hparams.norm_data, patch_stats=patch_stats)

    np.save(results_dir / "target_original.npy", target_phys)
    np.save(results_dir / "target_corrupted.npy", target_corrupted_phys)
    np.save(results_dir / "reconstructed_synth.npy", final_synth_phys)

    logprint(f"[INFO] Volumi salvati con successo in: {results_dir}", verbose)

    # =========================================================================

    os.makedirs(hparams.output_dir_BRGM_ddim, exist_ok=True)
    torch.save(
        {
            "epoch": hparams.num_steps,
            "latent_variable": latent_variable,
            "cond": cond,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),  # Salvataggio stato dello scheduler
        },
        f"{hparams.output_dir_BRGM_ddim}/checkpoint.pth",
    )

    return latent_variable_out, cond_out, {"loss": latest_metrics.get("loss", 0.0), "ssim": latest_metrics.get("ssim", 0.0)}


def main(hparams: Namespace) -> None:
    device = torch.device(hparams.device)
    
    writer = SummaryWriter(log_dir=hparams.tensor_board_logger_ddim)

    img_tensor, patch_stats = load_target_image(hparams, device=device)

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

    if img_tensor.ndim == 4:
        img_tensor = img_tensor.unsqueeze(0)
        
    diffusion, decoder = load_pre_trained_model(hparams, device=device)
    ddim = DDIMSampler(diffusion)
    
    forward = create_corruption_function(hparams=hparams, device=device)

    final_z, final_cond, _ = project(
        ddim, decoder, forward, img_tensor, device, writer, hparams, patch_stats, verbose=True
    )

    save_path = hparams.output_dir_BRGM_ddim
    torch.save({"z": final_z, "cond": final_cond}, f"{save_path}/results.pth")
    print(f"Risultati salvati in {save_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Inversione Diffusion Model per Dati Astrofisici")
    add_argument(parser)
    args = parser.parse_args()
    main(args)