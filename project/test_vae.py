import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.dataset_v3 import RadioPatchDataset
from models.aekl_no_attention import AutoencoderKL
from utils.config_train import train_config
from utils.config_aekl_v3 import get_hparams

def test():
    # 1. Caricamento Configurazioni
    train_param, _ = train_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hparams, _ = get_hparams()
    # 2. Dataset di Test
    # Assicurati di passare i parametri corretti (data_dir e catalogue_path)
    test_dataset = RadioPatchDataset(
        data_dir=train_param.test_dir, 
        catalogue_path=train_param.catalogue_path, # Se hai un catalogo separato per il test
        in_channels=hparams.in_channels # Passa il numero di canali corretto (1 o 3) in base alla tua configurazione
    )
    
    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=2, 
        shuffle=False, 
        num_workers=4
    )

    # 3. Caricamento Modello
    # Sostituisci con il path del tuo checkpoint migliore
    checkpoint_path = os.path.join(train_param.vae_path)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    hparams_dict = vars(hparams)  # Converti Namespace in dict
    
    
    model = AutoencoderKL(embed_dim=hparams.z_channels, hparams=hparams_dict).to(device)
    # Se il checkpoint è un dizionario con i pesi diretti
    if isinstance(checkpoint, dict):
        # Se per caso avevi salvato con la chiave 'model_state_dict'
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Carica direttamente il dizionario (MLflow standard)
            model.load_state_dict(checkpoint)
    print("Modello inizializzato e pesi caricati con successo.")
    
    model.eval()

    # 4. Loop di Test
    test_recon_loss = []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader)):
            x = batch["x_0"].to(device)
            
            # Forward pass manuale (come in run_step)
            h = model.encoder(x)
            z = model.quant_conv_mu(h) # Usiamo il mean per il test (determinismo)
            x_hat = model.decode(z)
            
            loss = F.mse_loss(x_hat, x)
            test_recon_loss.append(loss.item())

            # Salva alcune immagini di confronto ogni tanto
            if i == 0:
                save_comparison(x, x_hat, "test_reconstruction_results.png")

    print(f"--- Risultati Test ---")
    print(f"Average MSE: {np.mean(test_recon_loss):.6f}")

def save_comparison(orig, recon, filename):
    n = min(orig.shape[0], 8) # Mostra fino a 8 immagini
    mid_idx = orig.shape[2] // 2
    
    fig, axes = plt.subplots(2, n, figsize=(20, 6))
    for i in range(n):
        axes[0, i].imshow(orig[i, 0, mid_idx].cpu(), cmap='hot')
        axes[0, i].set_title("Originale")
        axes[0, i].axis('off')
        
        axes[1, i].imshow(recon[i, 0, mid_idx].cpu(), cmap='hot')
        axes[1, i].set_title("Ricostruito")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Confronto visivo salvato in: {filename}")
    plt.close()

if __name__ == "__main__":
    test()