import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import mlflow 

from torch.utils.data import DataLoader
from utils.dataset_v3 import RadioPatchDataset
from utils.config_train import train_config
from utils.config_unet_v3 import get_config
from utils.config_aekl_v3 import get_hparams
from models.aekl_no_attention import AutoencoderKL
from models.ddpm_v2_conditioned import DDPM

from models.ddim import DDIMSampler
from utils.plot_new import denormalize_data, comparison_plots_ok
from utils.const import IMAGE_SHAPE

SCALE_FACTOR_VAE = 0.18215


def quick_test_metrics(model, vae, dataloader, train_param, hparams, unet_cfg):
    model.eval()
    vae.eval()
     
    os.makedirs(train_param.test_fig, exist_ok=True)
    sampler = DDIMSampler(model)
    device = train_param.device
    epoch = train_param.epochs
    
    mses, ssims, psnrs = [], [], []
    
    # Parametri di shape fissi
    latent_size = IMAGE_SHAPE[2] // 4
    shape = (hparams.z_channels, latent_size, latent_size, latent_size)
    ddim_steps = 50 
    use_mask_channel = unet_cfg["params"]["use_mask_channel"]  # Assicurati sia booleano dal tuo argparser
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            x_start = batch["x_0"].to(device)
            spatial_mask = batch.get("spatial_mask", None)
            if spatial_mask is not None:
                spatial_mask = spatial_mask.to(device)
                
            context = batch.get("context", None)
            if context is not None:
                context = context.to(device)

            # --- 1. ENCODING LATENTE VIA VAE ---
            h = vae.encoder(x_start)
            z = vae.quant_conv_mu(h)
            z = z * SCALE_FACTOR_VAE

            # --- 2. PREPARAZIONE MASK LATENT (Se abilitata) ---
            mask_latent = None
            if use_mask_channel and spatial_mask is not None:
                latent_shape = z.shape[2:]  # (D_lat, H_lat, W_lat)
                mask_latent = F.interpolate(
                    spatial_mask, 
                    size=latent_shape, 
                    mode='trilinear', 
                    align_corners=False
                )

            # --- 3. COSTRUZIONE COND_PAYLOAD ---
            cond_payload = {}
            if train_param.cond_key == "crossattn" and context is not None:
                cond_payload["c_crossattn"] = [context]

            if train_param.cond_key == "concat" and mask_latent is not None:
                cond_payload["c_concat"] = [mask_latent]

            # --- 4. SAMPLING CON DDIM ---
            current_batch_size = x_start.shape[0]
            _, _, D_lat, H_lat, W_lat = z.shape
            shape = (hparams.z_channels, D_lat, H_lat, W_lat)
            
            sampler = DDIMSampler(model)
            img_noise = torch.randn((current_batch_size, *shape), device=train_param.device)

            # Passa cond_payload (oppure None se cond_key == "None")
            conditioning = cond_payload if train_param.cond_key not in [None, "None", "none"] else None

            z_gen, _ = sampler.sample(
                S=ddim_steps,
                batch_size=current_batch_size,
                shape=shape,
                conditioning=conditioning,  # <-- Usa il payload corretto!
                first_img=img_noise,
                eta=0.0,
                verbose=False
            )

            # --- 5. DECODING VAE E METRICHE ---
            z_gen_unscaled = z_gen / SCALE_FACTOR_VAE
            x_gen = vae.decode(z_gen_unscaled)
            


            x_start_norm = denormalize_data(x_start, hparams)
            x_gen_norm = denormalize_data(x_gen, hparams)


            if batch_idx % 10 == 0 or batch_idx == len(dataloader) - 1:
                # 1. Recupera i nomi dal batch (se disponibili nel dizionario del dataset)
                # Esempio: batch["patch_name_x0"] e batch["patch_name_context"]
                name_real = batch.get("name_x0", [f"Patch_x0_{batch_idx}"])[0]
                name_cond = batch.get("name_context", [f"Patch_context_{batch_idx}"])[0]
                

                if train_param.cond_key in [None, "None", "none"]:
                    title_r = f"Real Target: {name_real}"
                    title_g = f"Unconditioned Sample (Epoch {train_param.epochs})"
                else:
                    name_cond = batch.get("name_context", [f"Patch_context_{batch_idx}"])[0]
                    title_r = f"Real Target: {name_real}"
                    title_g = f"Generated from: {name_cond}"
                
                # 2. Chiama la funzione aggiornata con i nomi
                fig = comparison_plots_ok(
                    x_start_norm, 
                    x_gen_norm, 
                    title_real=title_r, 
                    title_gen=title_g, 
                    flag='test'
                )
                
                fig.savefig(f"{train_param.test_fig}/{train_param.norm_mode}_{batch_idx}.png")
                plt.close(fig)
                print(f"Salvata figura di confronto per batch {batch_idx}")
            # 4. Ciclo su OGNI elemento del batch corrente
            for i in range(current_batch_size):
                # Portiamo in CPU/Numpy l'i-esimo elemento
                img_true = x_start_norm[i, 0].cpu().numpy()
                img_gen = x_gen_norm[i, 0].cpu().numpy()
                img_gen = np.clip(img_gen, 0, 1)
                
                # --- NOTA SULLE METRICHE 3D ---
                # Se i tuoi dati sono 3D (visto che usi z_channels, D, H, W), calcolare le metriche 
                # solo sulla slice centrale potrebbe essere riduttivo. 
                # Opzione A: Mantieni la slice centrale (veloce)
                mid = img_true.shape[0] // 2
                true_slice = img_true[mid]
                gen_slice = img_gen[mid]
                
                mses.append(mse(true_slice, gen_slice))
                ssims.append(ssim(true_slice, gen_slice, data_range=1.0))
                psnrs.append(psnr(true_slice, gen_slice, data_range=1.0))
                
                # Opzione B (Consigliata se hai tempo): Calcola SSIM/PSNR sull'intero volume 3D
                # Per farlo, skimage.metrics supporta volumi 3D se specifichi data_range.
                # mses.append(mse(img_true, img_gen))
                # ssims.append(ssim(img_true, img_gen, data_range=1.0))
                # psnrs.append(psnr(img_true, img_gen, data_range=1.0))

        # 5. Aggregazione finale di tutti gli elementi di tutti i batch
        metrics = {
            "test/mse": np.mean(mses),
            "test/ssim": np.mean(ssims),
            "test/psnr": np.mean(psnrs)
        }
        
        print(f"\n--- GLOBAL TEST METRICS EPOCH {epoch} ---")
        for k, v in metrics.items():
            print(f"{k}: {v:.6f}")
            try:
                mlflow.log_metric(k, v, step=epoch)
            except:
                pass
        
        return metrics

if __name__ == "__main__":
    # Esempio di utilizzo
  

    # Setup dataset e dataloader
    train_param, _ = train_config()
    hparams, _ = get_hparams()
    unet_cfg, _ = get_config()
    unet_cfg["params"]["in_channels"] = unet_cfg["params"]["in_channels_unet"]
    unet_cfg["params"]["out_channels"] = unet_cfg["params"]["out_channels_unet"]
    unet_cfg["params"].pop("out_channels_unet", None)  # Rimuoviamo i parametri specifici del config per evitare confusione
    unet_cfg["params"].pop("in_channels_unet", None)
    
    dataset = RadioPatchDataset(data_dir=train_param.test_dir, catalogue_path=train_param.catalogue_path, in_channels=hparams.in_channels, norm_mode=train_param.norm_mode)
    dataloader = DataLoader(dataset, batch_size=train_param.batch_size, shuffle=True, num_workers=1, pin_memory=True, persistent_workers=True)

    # Inizializza VAE e DDPM 
    vae = AutoencoderKL(embed_dim=hparams.z_channels, hparams=vars(hparams)).to(train_param.device)
    vae_path = train_param.vae_path
    
    if os.path.exists(vae_path):
        vae_checkpoint = torch.load(vae_path, map_location=train_param.device, weights_only=True)
        
        if isinstance(vae_checkpoint, dict) and "model_state_dict" in vae_checkpoint:
            state_dict = vae_checkpoint["model_state_dict"]
            print("Estratto 'model_state_dict' dal checkpoint globale.")
        else:
            state_dict = vae_checkpoint
        vae.load_state_dict(state_dict)
        print(f"Pesi del VAE caricati con successo da {vae_path}!")
    else:
        print(f"ATTENZIONE: Checkpoint VAE non trovato in {vae_path}!")
    

    model = DDPM(
        unet_config=unet_cfg,
        conditioning_key=train_param.cond_key, 
        learn_logvar=True
    ).to(train_param.device)
    ddpm_path = train_param.output_dir_ddpm
    if os.path.exists(ddpm_path):
        ddpm_checkpoint = torch.load(ddpm_path, map_location=train_param.device, weights_only=True)
        # Controlla se i pesi sono dentro 'model_state_dict' (come dice l'errore)
        if isinstance(ddpm_checkpoint, dict) and "model_state_dict" in ddpm_checkpoint:
            state_dict = ddpm_checkpoint["model_state_dict"]
            print("Estratto 'model_state_dict' dal checkpoint globale.")
        else:
            state_dict =ddpm_checkpoint
        # Adatta in base a come salvi il dizionario di stato (es. checkpoint['model_state_dict'])
        model.load_state_dict(state_dict)
        print(f"Pesi del DDPM caricati con successo da {ddpm_path}!")
    else:
        print(f"ATTENZIONE: Checkpoint DDPM non trovato in {ddpm_path}!")
    # Esegui il test rapido
    quick_test_metrics(model, vae, dataloader, train_param, hparams=hparams, unet_cfg=unet_cfg)