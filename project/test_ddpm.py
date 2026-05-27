import torch
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

@torch.no_grad()
def quick_test_metrics(model, vae, dataloader, train_param, hparams, max_batches=None):
    model.eval()
    vae.eval()
    
    sampler = DDIMSampler(model)
    device = train_param.device
    epoch = train_param.epochs
    
    mses, ssims, psnrs = [], [], []
    
    # Parametri di shape fissi
    latent_size = IMAGE_SHAPE[2] // 4
    shape = (hparams.z_channels, latent_size, latent_size, latent_size)
    ddim_steps = 50 
    
    print(f"Inizio valutazione su {len(dataloader) if max_batches is None else max_batches} batch...")
    
    # 1. Ciclo su tutto il dataloader
    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
            
        x_start = batch["x_0"].to(device)
        context = batch["context"].to(device)
        current_batch_size = x_start.shape[0] # Gestisce anche l'ultimo batch se più piccolo
        
        # Generazione rumore dinamica in base al batch size corrente
        img_noise = torch.randn((current_batch_size, *shape), device=device)
        
        # 2. Campionamento
        z_gen, _ = sampler.sample(
            S=ddim_steps,
            batch_size=current_batch_size,
            shape=shape,
            conditioning=context,
            first_img=img_noise,
            eta=0.0, 
            verbose=False
        )
        
        # 3. Decodifica e Denormalizzazione
        x_gen = vae.decode(z_gen)
        if not isinstance(x_gen, torch.Tensor):
            x_gen = x_gen.sample()
            
        x_start_norm = denormalize_data(x_start, hparams)
        x_gen_norm = denormalize_data(x_gen, hparams)
        
        if batch_idx % 50 == 0 or batch_idx == len(dataloader) - 1: # Salva alcune figure di confronto ogni 50 batch
            fig = comparison_plots_ok(x_start_norm, x_gen_norm, flag='test')
            fig.savefig(f"{train_param.test_fig}/{train_param.norm_mode}_{batch_idx}.png")
            plt.close(fig) # Chiudi la figura per non consumare memoria
            print(f"Salvata figura di confronto per batch {batch_idx} (norm_mode={train_param.norm_mode})")
        
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
    
    dataset = RadioPatchDataset(data_dir=train_param.data_dir, catalogue_path=train_param.catalogue_path, in_channels=hparams.in_channels, norm_mode=train_param.norm_mode)
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
    quick_test_metrics(model, vae, dataloader, train_param, hparams=hparams)