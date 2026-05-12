import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.dataset_v3 import RadioPatchDataset
from models.aekl_no_attention import AutoencoderKL
from utils.config_train import train_config
from utils.config_aekl_v3 import get_hparams
from utils.plot_new import denormalize_data
from ml_flow_train_vae_decoder import comparison_plots_ok


def test():
    # 1. Caricamento Configurazioni
    train_param, _ = train_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Stai usando il dispositivo: {device}")
    hparams, _ = get_hparams()

    # 2. Dataset di Test
    test_dataset = RadioPatchDataset(
        data_dir=train_param.test_dir, 
        catalogue_path=train_param.catalogue_path,
        in_channels=hparams.in_channels 
    )
    
    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=2, 
        shuffle=False, 
        num_workers=1, # Aumentato per Leonardo
        pin_memory=True if torch.cuda.is_available() else False
    )

    # 3. Caricamento Modello
    checkpoint_path = os.path.join(train_param.vae_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    hparams_dict = vars(hparams)
    model = AutoencoderKL(embed_dim=hparams.z_channels, hparams=hparams_dict).to(device)
    
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    print("Modello inizializzato e pesi caricati con successo.")
    
    model.eval()

    # 4. Loop di Test
    test_recon_loss = []
    
    # Assicurati che la cartella esista
    os.makedirs("./job_script", exist_ok=True)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader)):
            x = batch["x_0"].to(device)
            
            # Forward pass
            h = model.encoder(x)
            z = model.quant_conv_mu(h)
            x_hat = model.decode(z)
            
            loss = F.mse_loss(x_hat, x)
            test_recon_loss.append(loss.item())

            # Salvataggio plot ogni 20 batch o all'ultimo
            if i % 20 == 0 or i == len(test_loader) - 1:
                # Applichiamo denormalizzazione per il plot
                x_plot = denormalize_data(x, train_param.norm_mode)
                x_hat_plot = denormalize_data(x_hat, train_param.norm_mode)

                fig = comparison_plots_ok(
                    x_plot, 
                    x_hat_plot, 
                    epoch=i,
                )
                fig.savefig(f"./job_script/test_vae/test_vae_recon_batch_{i}.png")
            
            # Fondamentale per non saturare la RAM
            del x, x_hat, h, z

    print(f"--- Risultati Test ---")
    print(f"Average MSE: {np.mean(test_recon_loss):.6f}")



if __name__ == "__main__":
    test()