# Code is adapted from: https://huggingface.co/spaces/Warvito/diffusion_brain/blob/main/app.py and
# https://colab.research.google.com/drive/1xJAor6_Ky36gxIk6ICNP--NMBjSlYKil?usp=sharing#scrollTo=4XDeCy-Vj59b
# Reference:
# [1] Pinaya, W. H., et al. (2022). "Brain Imaging Generation with Latent Diffusion Models." arXiv preprint arXiv:2209.07162.
# [2] Marinescu, R., et al. (2020). Bayesian Image Reconstruction using Deep Generative Models.

import math
import os
import csv
from argparse import ArgumentParser, Namespace
from time import perf_counter
from typing import List, Tuple
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import normalized_root_mse as nmse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from sklearn.neighbors import NearestNeighbors
from torch.utils.tensorboard import SummaryWriter

from models.BRGM.forward_models import ForwardAbstract
from utils.add_argument import add_argument
from utils.const import INPUT_FOLDER_CAT
from utils.plot_new import (
    draw_corrupted_images, 
    draw_images, 
    draw_img, 
    compare_cubes, 
    plot_orthogonal_cuts, 
    comparison_plots_ok, 
    denormalize_data
)
from utils.utils_new import (
    create_corruption_function,
    generating_latent_vector,
    getVggFeatures,
    inference,
    load_ddpm_latent_vectors,
    load_ddpm_model,
    load_pre_trained_decoder,
    load_target_image,
    load_vgg_perceptual,
    setup_noise_inputs,
)


def logprint(message: str, verbose: bool) -> None:
    if verbose:
        print(message)

def get_val(v):
    return v.detach().item() if hasattr(v, 'detach') else v

def add_hparams_to_tensorboard(
    hparams: Namespace,
    metrics: dict,
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
        "metrics/mse": metrics["mse"]
    }
    
    writer.add_hparams(hparam_dict, metric_dict)


def compute_latent_vector_stats(
    latent_vectors: torch.Tensor,
    device: torch.device,
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    logprint("Computing latent vector stats", verbose)
    latent_mean = torch.mean(latent_vectors, axis=0, keepdim=True)
    latnet_std = torch.std(latent_vectors, dim=0, keepdim=True, unbiased=False)
    return latent_mean, latnet_std


def compute_prior_loss(
    cur_latent_vector: torch.Tensor,
    latent_vectors: torch.Tensor,
    latent_vector_std: torch.Tensor,
    knn_model: NearestNeighbors,
    hparams: Namespace,
) -> Tuple[torch.Tensor, List[int]]:
    cur_latent_vector_np = cur_latent_vector.detach().cpu().numpy().reshape((1, -1))
    _, indices = knn_model.kneighbors(cur_latent_vector_np, n_neighbors=hparams.k)
    nearest_latent_vectors = latent_vectors[indices[0]]
    mean_nearest_latent_vector = torch.mean(
        nearest_latent_vectors, axis=0, keepdim=True
    )
    prior_loss = (
        (
            (cur_latent_vector / latent_vector_std)
            - (mean_nearest_latent_vector / latent_vector_std)
        )
        .abs()
        .mean()
    )
    return prior_loss, indices[0]


def project(
    vqvae: torch.nn.Module,
    forward: ForwardAbstract,  # Corruption function
    target: torch.Tensor,
    device: torch.device,
    writer: SummaryWriter,
    hparams: Namespace,
    patch_stats: dict = None, # <-- AGGIUNTO: Per trasferire le statistiche dinamiche di scala
    verbose: bool = False,
):
    latent_vectors_tensor = load_ddpm_latent_vectors(device, hparams)
    latent_vector_mean, latent_vector_std = compute_latent_vector_stats(
        latent_vectors=latent_vectors_tensor, device=device, verbose=verbose
    )

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
    cond, latent_variable = setup_noise_inputs(cat, device=device, hparams=hparams)
    cond_crossatten = cond.unsqueeze(1)
    cond_concat = cond.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    cond_concat = cond_concat.expand(list(cond.shape[0:2]) + list(hparams.image_size))
    
    if not hparams.mean_latent_vector:
        ddpm = load_ddpm_model(ddpm_path=hparams.path_to_ddpm_checkpoint, device=device)        
        conditioning = {
            "c_concat": [cond_concat.float().to(device)],
            "c_crossattn": [cond_crossatten.float().to(device)],
        }
        with torch.no_grad():
            latent_vector = generating_latent_vector(
                diffusion=ddpm,
                latent_variable=latent_variable,
                conditioning=conditioning,
                batch_size=1,
                image_size=hparams.image_size,
                scale_factor=hparams.downsample_factor,
                z_channels=hparams.z_channels,
            )
    else:
        latent_vector = latent_vector_mean.clone().detach()
    latent_vector.requires_grad = True

    update_params = [latent_vector]

    optimizer_adam = torch.optim.Adam(
        update_params,
        betas=(0.9, 0.999),
        lr=hparams.learning_rate,
    )
    latent_vector_out = torch.zeros(
        [hparams.num_steps] + list(latent_vector.shape[1:]),
        dtype=torch.float32,
        device=device,
    )

    target_img_corrupted = forward(target)
    vgg16, target_features = load_vgg_perceptual(hparams, target_img_corrupted, device)
    total_num_pixels = (
        target_img_corrupted.numel()
        if hparams.corruption != "mask"
        else math.prod(forward.mask.shape) - forward.mask.sum()
    )

    save_path = Path(hparams.output_dir_BRGM_decoder)
    save_path.mkdir(parents=True, exist_ok=True)

    for step in range(hparams.start_steps, hparams.num_steps):

        def closure():
            optimizer_adam.zero_grad()

            synth_img = inference(vqvae=vqvae, latent_vectors=latent_vector)
            synth_img_corrupted = forward(synth_img)

            loss = 0
            downsampling_loss = 0
            
            if hparams.corruption != "None":
                pixelwise_loss = (synth_img_corrupted - target_img_corrupted).abs().sum() / total_num_pixels
                loss += pixelwise_loss

                synth_features = getVggFeatures(hparams, synth_img_corrupted, vgg16)
                perc_loss = (target_features - synth_features).abs().mean()
                loss += hparams.lambda_perc * perc_loss
            else:
                pixelwise_loss = (synth_img - target).abs().mean()
                loss += (1 - hparams.alpha_downsampling_loss) * pixelwise_loss

                synth_features = getVggFeatures(hparams, synth_img_corrupted, vgg16)
                perc_loss = (target_features - synth_features).abs().mean()
                loss += hparams.lambda_perc * perc_loss

            loss.backward(retain_graph=True)

            synth_img_np = synth_img[0, 0].detach().cpu().numpy()
            target_np = target[0, 0].detach().cpu().numpy()
            
            # --- FIX: Passiamo patch_stats a denormalize_data se presenti ---
            synth_m = denormalize_data(synth_img_np, norm_mode=hparams.norm_data, patch_stats=patch_stats)
            target_m = denormalize_data(target_np, norm_mode=hparams.norm_data, patch_stats=patch_stats)

            data_range = float(np.max([synth_m.max(), target_m.max()]) - np.min([synth_m.min(), target_m.min()]))

            if data_range == 0:
                data_range = 1.0

            ssim_ = ssim(
                synth_m, 
                target_m, 
                win_size=11, 
                data_range=data_range, 
                gaussian_weights=True, 
                use_sample_covariance=False
            )
            
            psnr_ = psnr(target_m, synth_m, data_range=data_range)
            mse_ = mse(target_m, synth_m)
            nmse_ = nmse(target_m, synth_m)

            # Scrittura scalari TensorBoard
            writer.add_scalar("loss", loss, global_step=step)
            writer.add_scalar("pixelwise_loss", pixelwise_loss, global_step=step)
            writer.add_scalar("perceptual_loss", perc_loss, global_step=step)
            writer.add_scalar("downsampling_loss", downsampling_loss, global_step=step)
            writer.add_scalar("ssim", ssim_, global_step=step)
            writer.add_scalar("psnr", psnr_, global_step=step)
            writer.add_scalar("mse", mse_, global_step=step)
            writer.add_scalar("nmse", nmse_, global_step=step)

            logprint(
                f"step {step + 1:>4d}/{hparams.num_steps}: tloss {get_val(loss)} pix_loss {get_val(pixelwise_loss)} perc_loss {get_val(perc_loss)}\n"
                f"              : SSIM {get_val(ssim_)} PSNR {get_val(psnr_)} MSE {get_val(mse_)}",
                verbose=verbose,
            )

            # PLOT INTERMEDI SU TENSORBOARD
            if step % 25 == 0:
                step_str = f"{step}".zfill(4)
                
                draw_img(
                    synth_m,
                    title=f"synth_step_{step_str}",
                    step=step_str,
                    output_folder=save_path,
                )
                
                if hparams.corruption != "None":
                    synth_corr_m = denormalize_data(synth_img_corrupted[0, 0].detach().cpu().numpy(), norm_mode=hparams.norm_data, patch_stats=patch_stats)
                    target_corr_m = denormalize_data(target_img_corrupted[0, 0].detach().cpu().numpy(), norm_mode=hparams.norm_data, patch_stats=patch_stats)
                    imgs = draw_corrupted_images(synth_m, target_m, synth_corr_m, target_corr_m, ssim_=ssim_)
                else:
                    imgs = draw_images(synth_m, target_m, ssim_=ssim_)
            
                writer.add_figure(f"step: {step_str}", imgs, global_step=step)
                plt.close(imgs)

            closure.final_metrics = {"loss": loss.item(), "ssim": ssim_, "psnr": psnr_, "mse": mse_, "nmse": nmse_}
            closure.final_synth = synth_img
            closure.final_synth_corr = synth_img_corrupted

            return loss
        
        torch.nn.utils.clip_grad_norm_([latent_vector], max_norm=1.0)
        optimizer_adam.step(closure=closure)
        latent_vector_out[step] = latent_vector.detach()[0]

    # --- FUORI DAL CICLO FOR: PLOT E LOG FINALI ---
    final_metrics = closure.final_metrics
    synth_img = closure.final_synth
    synth_img_corrupted = closure.final_synth_corr
    
    add_hparams_to_tensorboard(
        hparams, metrics=final_metrics,
        writer=writer,
    )

    # --- FIX: Denormalizzazione sicura con patch_stats ---
    synth_vis = denormalize_data(synth_img[0, 0].detach().cpu().numpy(), norm_mode=hparams.norm_data, patch_stats=patch_stats)
    target_vis = denormalize_data(target[0, 0].detach().cpu().numpy(), norm_mode=hparams.norm_data, patch_stats=patch_stats)
    target_img_corrupted_vis = denormalize_data(target_img_corrupted[0, 0].detach().cpu().numpy(), norm_mode=hparams.norm_data, patch_stats=patch_stats)
    synth_img_corrupted_vis = denormalize_data(synth_img_corrupted[0, 0].detach().cpu().numpy(), norm_mode=hparams.norm_data, patch_stats=patch_stats)

    # --- SANITIZZAZIONE NAN / INF ---
    synth_vis = np.nan_to_num(synth_vis, nan=0.0, posinf=1.0, neginf=0.0)
    target_vis = np.nan_to_num(target_vis, nan=0.0, posinf=1.0, neginf=0.0)
    synth_img_corrupted_vis = np.nan_to_num(synth_img_corrupted_vis, nan=0.0, posinf=1.0, neginf=0.0)
    target_img_corrupted_vis = np.nan_to_num(target_img_corrupted_vis, nan=0.0, posinf=1.0, neginf=0.0)

    # DEBUG PRINT
    print(f"TARGET - Min: {target_vis.min():.2e}, Max: {target_vis.max():.2e}, Mean: {target_vis.mean():.2e}")
    print(f"SYNTH  - Min: {synth_vis.min():.2e}, Max: {synth_vis.max():.2e}, Mean: {synth_vis.mean():.2e}")
    print("Synth ha NaN?:", np.isnan(synth_vis).any())

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
    writer.flush()
    writer.close()

    torch.save(
        {
            "epoch": hparams.num_steps - 1,
            "latent_vectors": latent_vector,
            "optimizer": optimizer_adam.state_dict(),
        },
        save_path / "checkpoint.pth",
    )

    print(f"Checkpoint saved to {save_path / 'checkpoint.pth'}")

    csv_results_path = Path("/leonardo_scratch/large/userexternal/gvitanza/InverseSR/data/decoder/result_decoder_downsample_2.csv")
    csv_results_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(csv_results_path, "a") as file:
        writer_csv = csv.writer(file)
        writer_csv.writerow([
            hparams.object_id,
            final_metrics["loss"],
            final_metrics["ssim"],
            final_metrics["psnr"],
            final_metrics["mse"],
            final_metrics["nmse"]
        ])

    return latent_vector_out


def main(hparams: Namespace) -> None:
    device = hparams.device
    
    # --- FIX: Estrazione se load_target_image restituisce anche le patch_stats ---
    target_data = load_target_image(hparams, device)
    if isinstance(target_data, tuple):
        img_tensor, patch_stats = target_data
    else:
        img_tensor, patch_stats = target_data, {}

    # Estraggo i dati e rimuovo dimensioni fittizie
    img_data = img_tensor.squeeze().cpu().numpy() # [Z, Y, X]

    moment_0 = np.sum(img_data, axis=0) 

    mid_slice = img_data.shape[0] // 2
    slice_to_plot = img_data[mid_slice]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    vmax_slice = np.percentile(slice_to_plot, 99.5)
    im0 = axes[0].imshow(slice_to_plot, origin="lower", cmap="inferno", vmax=vmax_slice)
    axes[0].set_title(f"Target Slice Z={mid_slice}")
    plt.colorbar(im0, ax=axes[0])

    vmax_mom0 = np.percentile(moment_0, 99.5)

    im1 = axes[1].imshow(moment_0, origin="lower", cmap="inferno", vmin=0, vmax=vmax_mom0)
    axes[1].set_title("Target - Momento 0 (Integrato su Z)")
    plt.colorbar(im1, ax=axes[1])

    output_path = Path(hparams.output_dir_BRGM_decoder) / "target_image_nel_main.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path)
    plt.close()
    
    print("Ha NaN?:", np.isnan(img_tensor.detach().cpu().numpy()).any())
    print("Min:", np.nanmin(img_tensor.detach().cpu().numpy()), "Max:", np.nanmax(img_tensor.detach().cpu().numpy()))
    
    if img_tensor.dim() == 4:
        img_tensor = img_tensor.unsqueeze(0)
        
    writer = SummaryWriter(log_dir=hparams.tensor_board_logger_decoder)

    forward = create_corruption_function(hparams=hparams, device=device)
    decoder = load_pre_trained_decoder(
        vae_path=hparams.decoder_path_BRGM,
        device=device,
    )

    start_time = perf_counter()
    latent_vector_out = project(
        decoder,
        writer=writer,
        hparams=hparams,
        forward=forward,
        target=img_tensor,
        device=device,
        patch_stats=patch_stats, # <-- Passato alle funzioni di plot
        verbose=True,
    )
    print(f"Elapsed: {(perf_counter() - start_time):.1f} s")

    torch.save(
        {"latent_vector_out": latent_vector_out},
        Path(hparams.output_dir_BRGM_decoder) / "latent_vector_out.pth",
    )

    print("Latent vector saved to", Path(hparams.output_dir_BRGM_decoder) / "latent_vector_out.pth")


if __name__ == "__main__":
    parser = ArgumentParser(description="Trainer args", add_help=False)
    add_argument(parser)
    hparams = parser.parse_args()
    main(hparams)