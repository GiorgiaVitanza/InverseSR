# Code is adpated from: https://huggingface.co/spaces/Warvito/diffusion_brain/blob/main/app.py and
# https://colab.research.google.com/drive/1xJAor6_Ky36gxIk6ICNP--NMBjSlYKil?usp=sharing#scrollTo=4XDeCy-Vj59b
# A lot of thanks to the author of the code
# Reference:
# [1] Pinaya, W. H., et al. (2022). "Brain Imaging Generation with Latent Diffusion Models." arXiv preprint arXiv:2209.07162.
# [2] Marinescu, R., et al. (2020). Bayesian Image Reconstruction using Deep Generative Models.

import math

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
    LATENT_SHAPE,
    OUTPUT_FOLDER,
    GLOBAL_MAX,
    GLOBAL_MIN,
    PRETRAINED_MODEL_DECODER_PATH,
)
from utils.plot_new import draw_corrupted_images, draw_images, draw_img, compare_cubes, plot_orthogonal_cuts
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

OUTPUT_FOLDER = OUTPUT_FOLDER / "BRGM_decoder"

def denormalize_data(x):
    """Denormalizza da 0-1 a valori originali con clipping"""
    # 1. Recupera i valori fisici originali 
    v_min = GLOBAL_MIN
    v_max = GLOBAL_MAX
    x = x * (v_max - v_min) + v_min
    return np.clip(x, v_min, v_max)

def logprint(message: str, verbose: bool) -> None:
    if verbose:
        print(message)

def get_val(v):
    return v.detach().item() if hasattr(v, 'detach') else v

def add_hparams_to_tensorboard(
    hparams: Namespace,
    metrics: dict,
    cond1: torch.Tensor,
    cond2: torch.Tensor,
    cond3: torch.Tensor,
    cond4: torch.Tensor,
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
        "inv_cond/hi_size": cond1.item(),
        "inv_cond/line_flux_integral": cond2.item(),
        "inv_cond/i": cond3.item(),
        "inv_cond/w20": cond4.item(),
    }
    
    writer.add_hparams(hparam_dict, metric_dict)



def create_mask_for_backprop(hparams: Namespace, device: torch.device) -> torch.Tensor:
    mask_cond = torch.ones((1, 4), device=device)
    mask_cond[:, 0] = 0 if not hparams.update_hi_size else 1
    mask_cond[:, 1] = 0 if not hparams.update_line_flux_integral else 1
    mask_cond[:, 2] = 0 if not hparams.update_i else 1
    mask_cond[:, 3] = 0 if not hparams.update_w20 else 1
    return mask_cond


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

    cat_path = Path("./data/inputs/128x128x128_stride128/master_patch_catalog.csv")
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
    cond_concat = cond_concat.expand(list(cond.shape[0:2]) + list(LATENT_SHAPE[2:]))
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
            )
    else:
        latent_vector = latent_vector_mean.clone().detach()
    latent_vector.requires_grad = True

    update_params = []
    update_params.append(latent_vector)

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

    # Compute latent representation stats.
    for step in range(hparams.start_steps, hparams.num_steps):

        def closure():
            optimizer_adam.zero_grad()

            synth_img = inference(
                vqvae=vqvae,
                latent_vectors=latent_vector,
            )
            synth_img_corrupted = forward(synth_img)

            loss = 0
            downsampling_loss = 0
            prior_loss = 0
            indices = [0]
            if hparams.corruption != "None":
                pixelwise_loss = (
                    synth_img_corrupted - target_img_corrupted
                ).abs().sum() / total_num_pixels
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

            return (
                loss,
                pixelwise_loss,
                perc_loss,
                downsampling_loss,
                prior_loss,
                synth_img,
                synth_img_corrupted,
                indices,
            )

        (
            loss,
            pixelwise_loss,
            perc_loss,
            downsampling_loss,
            prior_loss,
            synth_img,
            synth_img_corrupted,
            indices,
        ) = optimizer_adam.step(closure=closure)

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
        writer.add_scalar("pixelwise_loss", pixelwise_loss, global_step=step)
        writer.add_scalar("perceptual_loss", perc_loss, global_step=step)
        writer.add_scalar("downsampling_loss", downsampling_loss, global_step=step)
        if prior_loss != 0:
            writer.add_scalar("prior_loss", prior_loss, global_step=step)
            writer.add_scalar("indice", indices[0], global_step=step)
        writer.add_scalar("ssim", ssim_, global_step=step)
        writer.add_scalar("psnr", psnr_, global_step=step)
        writer.add_scalar("mse", mse_, global_step=step)
        writer.add_scalar("nmse", nmse_, global_step=step)

        logprint(
            f"step {step + 1:>4d}/{hparams.num_steps}: tloss {get_val(loss)} pix_loss {get_val(pixelwise_loss)} perc_loss {get_val(perc_loss)} prior_loss {get_val(prior_loss)}\n"
            f"              : SSIM {get_val(ssim_)} PSNR {get_val(psnr_)} MSE {get_val(mse_)} NMSE {get_val(nmse_)}",
            verbose=verbose,
        )

        step_ = f"{step}".zfill(4)
        draw_img(
            synth_img_np,
            title="synth",
            step=step_,
            output_folder=OUTPUT_FOLDER,
        )

        if step % 25 == 0:
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

        latent_vector_out[step] = latent_vector.detach()[0]

    add_hparams_to_tensorboard(
        hparams,
        metrics={"loss": loss.item(), "ssim": ssim_, "psnr": psnr_, "mse": mse_, "nmse": nmse_},
        cond1=cond_concat[0, 0, 0, 0, 0].cpu(),
        cond2=cond_concat[0, 1, 0, 0, 0].cpu(),
        cond3=cond_concat[0, 2, 0, 0, 0].cpu(),
        cond4=cond_concat[0, 3, 0, 0, 0].cpu(),
        writer=writer,
    )

    # --- ESEMPIO DI MODIFICA PRIMA DEI PLOT FINALI ---

    
    # 2. De-normalizza l'immagine sintetica (che è in [0, 1]) 
    # per portarla nella scala fisica del target
    synth_vis = synth_img[0, 0].detach().cpu().numpy()
    synth_vis = denormalize_data(synth_vis)
    target_vis = target[0, 0].detach().cpu().numpy() 
    target_vis = denormalize_data(target_vis)
    target_img_corrupted_vis = target_img_corrupted[0, 0].detach().cpu().numpy() 
    target_img_corrupted_vis = denormalize_data(target_img_corrupted_vis)
    synth_img_corrupted_vis = synth_img_corrupted[0, 0].detach().cpu().numpy() 
    synth_img_corrupted_vis = denormalize_data(synth_img_corrupted_vis)

    draw_img(
        target_np,
        title="target",
        step=step_,
        output_folder=OUTPUT_FOLDER,
    )

    draw_img(
        synth_img_corrupted[0, 0].detach().cpu().numpy(),
        title="corrupted",
        step=step_,
        output_folder=OUTPUT_FOLDER,
    )

    compare_cubes(
        target_vis,
        synth_vis,
        title="target_vs_synth",
        save_path=OUTPUT_FOLDER / "compare_target_vs_synth.png",
    )

    compare_cubes(
        target_img_corrupted_vis,
        synth_img_corrupted_vis,
        title="corrupted_target_vs_corrupted_synth",
        save_path=OUTPUT_FOLDER / "compare_corrupted_target_vs_corrupted_synth.png",
    )

    plot_orthogonal_cuts(
        synth_vis,
        title="orthogonal_cuts_synth",
        save_path=OUTPUT_FOLDER/"orthogonal_cuts_synth.png",
    )

    plot_orthogonal_cuts(
        target_vis,
        title="orthogonal_cuts_target", 
        save_path=OUTPUT_FOLDER/"orthogonal_cuts_target.png",
    )

    plot_orthogonal_cuts(
        synth_img_corrupted_vis,
        title="orthogonal_cuts_corrupted",
        save_path=OUTPUT_FOLDER/"orthogonal_cuts_corrupted.png",
    )

    writer.flush()
    writer.close()

    torch.save(
        {
            "epoch": step,
            "latent_vectors": latent_vector,
            "optimizer": optimizer_adam.state_dict(),
        },
        OUTPUT_FOLDER / "checkpoint.pth",
    )

    row = [
        hparams.object_id,
        ssim_,
        psnr_,
        mse_,
        nmse_,
    ]

    with open(
        "./data/decoder/result_decoder_downsample_2.csv",
        "a",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(row)

    return latent_vector_out


def main(hparams: Namespace) -> None:
    # device = torch.device("cuda" if COMPUTECANADA else "cpu")
    # Don't have enough memory to run on GPU. :(
    device = hparams.device
    img_tensor = load_target_image(hparams, device)
    writer = SummaryWriter(log_dir=hparams.tensor_board_logger)

    forward = create_corruption_function(hparams=hparams, device=device)
    decoder = load_pre_trained_decoder(
        vae_path=PRETRAINED_MODEL_DECODER_PATH,
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
        OUTPUT_FOLDER / "latent_vector_out.pth",
    )


if __name__ == "__main__":
    parser = ArgumentParser(description="Trainer args", add_help=False)
    add_argument(parser)
    hparams = parser.parse_args()
    # seed_everything(hparams.seed)
    main(hparams)
