# Code is adpated from: https://huggingface.co/spaces/Warvito/diffusion_brain/blob/main/app.py and
# https://colab.research.google.com/drive/1xJAor6_Ky36gxIk6ICNP--NMBjSlYKil?usp=sharing#scrollTo=4XDeCy-Vj59b
# A lot of thanks to the author of the code
# Reference:
# [1] Pinaya, W. H., et al. (2022). "Brain Imaging Generation with Latent Diffusion Models." arXiv preprint arXiv:2209.07162.
# [2] Marinescu, R., et al. (2020). Bayesian Image Reconstruction using Deep Generative Models.

import math
import os

# from joblib import dump, load
from argparse import ArgumentParser, Namespace
from time import perf_counter
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
from models.BRGM.forward_models import ForwardAbstract
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import normalized_root_mse as nmse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from sklearn.neighbors import NearestNeighbors
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

from utils.add_argument import add_argument
from utils.const import (
    FITS_LIMIT,
    FITS_MEAN,
    FITS_STD,
    PRETRAINED_MODEL_DECODER_PATH,
    INPUT_FOLDER_CAT
)
from utils.plot_new import draw_corrupted_images, draw_images, draw_img, compare_cubes, plot_orthogonal_cuts, comparison_plots_ok, denormalize_data
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

    # Definizione sicura dei percorsi all'inizio della funzione per evitare NameError
    save_path = Path(hparams.output_dir_BRGM_decoder)
    save_path.mkdir(parents=True, exist_ok=True)

    # Compute latent representation stats.
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
            
            # Dati denormalizzati per metriche e plot stabili
            synth_m = denormalize_data(synth_img_np, norm_mode=hparams.norm_data)
            target_m = denormalize_data(target_np, norm_mode=hparams.norm_data)

            ssim_ = ssim(synth_m, target_m, win_size=11, data_range=1.0, gaussian_weights=True, use_sample_covariance=False)
            data_range = np.max([synth_m.max(), target_m.max()]) - np.min([synth_m.min(), target_m.min()])
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

            # PLOT INTERMEDI SU TENSORBOARD (Ogni 25 step)
            if step % 25 == 0:
                step_str = f"{step}".zfill(4)
                
                # Salviamo l'immagine includendo lo step nel titolo per non sovrascriverla continuamente
                draw_img(
                    synth_m,
                    title=f"synth_step_{step_str}",
                    step=step_str,
                    output_folder=save_path,
                )
                
                if hparams.corruption != "None":
                    # Usiamo i dati denormalizzati per coerenza visiva su TensorBoard
                    synth_corr_m = denormalize_data(synth_img_corrupted[0, 0].detach().cpu().numpy(), norm_mode=hparams.norm_data)
                    target_corr_m = denormalize_data(target_img_corrupted[0, 0].detach().cpu().numpy(), norm_mode=hparams.norm_data)
                    imgs = draw_corrupted_images(synth_m, target_m, synth_corr_m, target_corr_m, ssim_=ssim_)
                else:
                    imgs = draw_images(synth_m, target_m, ssim_=ssim_)
            
                writer.add_figure(f"step: {step_str}", imgs, global_step=step)
                plt.close(imgs)

            # Salvataggio metriche nell'oggetto closure per l'esterno
            closure.final_metrics = {"loss": loss.item(), "ssim": ssim_, "psnr": psnr_, "mse": mse_, "nmse": nmse_}
            closure.final_synth = synth_img
            closure.final_synth_corr = synth_img_corrupted

            return loss
        
        optimizer_adam.step(closure=closure)
      
        latent_vector_out[step] = latent_vector.detach()[0]

    # --- FUORI DAL CICLO FOR: PLOT E LOG FINALI ---
    final_metrics = closure.final_metrics
    synth_img = closure.final_synth
    synth_img_corrupted = closure.final_synth_corr
    
    # Registrazione iperparametri finale
    add_hparams_to_tensorboard(
        hparams, metrics=final_metrics,
        writer=writer,
    )

    # Denormalizzazione totale per i plot di chiusura (Scala fisica Jy/beam)
    synth_vis = denormalize_data(synth_img[0, 0].detach().cpu().numpy(), hparams.norm_data)
    target_vis = denormalize_data(target[0, 0].detach().cpu().numpy(), hparams.norm_data)
    target_img_corrupted_vis = denormalize_data(target_img_corrupted[0, 0].detach().cpu().numpy(), hparams.norm_data)
    synth_img_corrupted_vis = denormalize_data(synth_img_corrupted[0, 0].detach().cpu().numpy(), hparams.norm_data)

    print(f"TARGET - Min: {target_vis.min():.2e}, Max: {target_vis.max():.2e}, Mean: {target_vis.mean():.2e}")
    print(f"SYNTH  - Min: {synth_vis.min():.2e}, Max: {synth_vis.max():.2e}, Mean: {synth_vis.mean():.2e}")

    # Stringa di safe finale per i nomi dei file
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
            "epoch": step,
            "latent_vectors": latent_vector,
            "optimizer": optimizer_adam.state_dict(),
        },
        save_path / "checkpoint.pth",
    )

    print(f"Checkpoint saved to {save_path / 'checkpoint.pth'}")


    with open(
        "/leonardo_scratch/large/userexternal/gvitanza/InverseSR/data/decoder/result_decoder_downsample_2.csv",
        "a",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(row)

    return latent_vector_out


def main(hparams: Namespace) -> None:
    # device = torch.device("cuda" if COMPUTECANADA else "cpu")
    # Don't have enough memory to run on GPU. :(
    device = hparams.device
    img_tensor = load_target_image(hparams, device) # carico il target normalizzato


    # 1. Prepara il dato (estrai la slice centrale del volume 3D)
    # img_tensor[0, 0] è [D, H, W]
    img_data = img_tensor[0].cpu().numpy()
    mid_slice = img_data.shape[0] // 2
    slice_to_plot = img_data[mid_slice]

    # 2. Crea il plot
    plt.figure(figsize=(8, 8))
    plt.imshow(slice_to_plot, cmap='hot') 
    plt.colorbar(label='Intensità')
    plt.title(f"Target image nel main normalizzato {hparams.norm_data} (Slice centrale)")

    # 3. Gestione salvataggio
    output_path = Path(hparams.output_dir_BRGM_decoder) / "target_image_nel_main.png"
    output_path.parent.mkdir(parents=True, exist_ok=True) # Crea la cartella se non esiste

    plt.savefig(output_path)
    plt.show() # Opzionale, se sei in un notebook
    plt.close() # Importante per liberare memoria
    
    if img_tensor.dim() == 4:  # Se manca la dimensione del batch, aggiungila
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
    # seed_everything(hparams.seed)
    main(hparams)
