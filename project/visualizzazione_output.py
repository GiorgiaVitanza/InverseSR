import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

import imageio

# Risale di una cartella rispetto a dove si trova questo script
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

# Assicurati di importare le tue classi dal file dove sono definite
from models.aekl_no_attention import AutoencoderKL 
from utils.plot_new import draw_img_in_three_dim
from utils.utils_new import generating_latent_vector
from utils.const import LATENT_SHAPE
from data.visualizzazione_3d import preprocess, animate_slices, static_grid, volume_rendering, isosurface

def visualize_reconstruction(checkpoint_path, model_path, output_path, flag):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Configurazione Hparams per il Decoder (deve corrispondere a quelli del training)
    # Questi sono valori tipici per l'architettura LDM Astro v2
    hparams = {
        "in_channels": 1,
        "n_channels": 64,
        "z_channels": 3,
        "ch_mult": [1, 2, 2], 
        "num_res_blocks": 2,
        "resolution": (160, 224, 160), # Esempio di risoluzione medica
        "attn_resolutions": [],
        "out_channels": 3,
    }
    embed_dim = 3 # Coerente con z_channels

    # 2. Caricamento del Modello
    print("Caricamento del decoder...")
    model = AutoencoderKL(embed_dim=embed_dim, hparams=hparams)
    
    # Carichiamo il VAE completo
    model = torch.load(model_path, map_location=device, weights_only=False)
    
    model.to(device).eval()

    # 3. Caricamento del Vettore Latente ottimizzato
    print(f"Caricamento del vettore latente da {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Il tuo codice salvava {'latent_vectors': ...}
    if flag == "ddim":    
        z_noisy = checkpoint['latent_variable'] 
        cond = checkpoint['cond']
        cond_crossatten = cond.unsqueeze(1)
        cond_concat = cond.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        cond_concat = cond_concat.expand(list(cond.shape[0:2]) + list(LATENT_SHAPE[2:]))
        conditioning_ottimizzato = {
            "c_concat": [cond_concat.float().to(device)],
            "c_crossattn": [cond_crossatten.float().to(device)],
        }
    
        
        
        # Carichiamo il checkpoint
        path_oggetto = Path("./data/trained_models_astro/ddpm/data/model.pth")
        ddpm = torch.load(path_oggetto, weights_only=False, map_location=device)
        
        
        # 2. Esegui il campionamento deterministico DDIM per ottenere z_0
        # Questo trasforma il rumore ottimizzato in un latente strutturato
        print("Esecuzione DDIM Sampling...")
        z = generating_latent_vector(
            diffusion=ddpm,
            latent_variable=z_noisy,
            conditioning=conditioning_ottimizzato, # Quello salvato nel checkpoint
            batch_size=1
        )
    elif flag == "decoder":
        z = checkpoint['latent_vectors']
    else:
        raise ValueError(f"Flag non riconosciuto: {flag}. Usa 'ddim' o 'decoder'.")
        

    
    # Assicuriamoci che z abbia la forma corretta [Batch, Channel, D, H, W]
    if z.dim() == 4:
        z = z.unsqueeze(0)

    # 4. Ricostruzione 3D
    print("Generazione volume 3D...")
    with torch.no_grad():
        reconstructed_volume = model.reconstruct_ldm_outputs(z)
        # Portiamo in formato numpy [D, H, W]
        rec_np = reconstructed_volume.squeeze().cpu().numpy()

    # 5. Visualizzazione con le tue funzioni
    print("Salvataggio delle sezioni ortogonali...")
    ## Riduciamo rec_np a 3 dimensioni in modo aggressivo
    if rec_np.ndim == 4:
        # Se ha 4 dimensioni (es. 3 canali), prendiamo solo il primo
        rec_np_3d = rec_np[0] 
    else:
        # Altrimenti rimuoviamo eventuali dimensioni 1 residue
        rec_np_3d = np.squeeze(rec_np)

    # Verifica di sicurezza: se è ancora 4D (es. era 5D all'inizio), prendi il primo elemento
    while rec_np_3d.ndim > 3:
        rec_np_3d = rec_np_3d[0]
    # Converti in Tensor e assicurati che sia 3D
    rec_tensor = torch.from_numpy(rec_np_3d)
    
    # Debug opzionale per essere sicuri:
    print(f"DEBUG: Tensor shape finale per plot: {rec_tensor.shape}")
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    
    draw_img_in_three_dim(rec_tensor, title="ricostruzione_finale", output_folder=out_dir)
    
    print(f"Visualizzazione completata! Trovi i file .png in: {out_dir}")
    return rec_np


def salva_datacube_gif(volume, output_path, axis=0, fps=15, use_log=True):
    """
    Versione ottimizzata per dati Radioastronomici:
    - Scaling logaritmico robusto
    - Normalizzazione globale con percentili (niente sfarfallio)
    - Colormap 'inferno' applicata correttamente
    """
    # 1. Pulizia dimensioni
    volume = np.squeeze(volume)
    if volume.ndim == 4:
        volume = volume[0] 

    # 2. Pre-processing Logaritmico (opzionale ma fondamentale per Astro)
    if use_log:
        # Portiamo il minimo a 0 per evitare log di numeri negativi
        # Aggiungiamo epsilon per evitare log(0)
        v_min_raw = volume.min()
        volume = np.log10(volume - v_min_raw + 1e-8)

    # 3. Calcolo limiti GLOBALI (Percentili) per coerenza tra i frame
    # Usiamo i percentili per ignorare outlier e rumore estremo
    v_min = np.percentile(volume, 1)    
    v_max = np.percentile(volume, 99.9) 

    # Otteniamo la colormap
    cmap = plt.get_cmap('inferno')
    frames_rgb = []

    # 4. Generazione Frame lungo l'asse scelto
    num_fette = volume.shape[axis]
    
    for i in range(num_fette):
        # Selezione della fetta in base all'asse
        if axis == 0:
            fetta = volume[i, :, :]
        elif axis == 1:
            fetta = volume[:, i, :]
        else:
            fetta = volume[:, :, i]
        
        # 5. Normalizzazione e Clipping (0.0 - 1.0)
        # Importante: clippiamo prima di scalare per evitare overflow
        fetta_norm = np.clip(fetta, v_min, v_max)
        fetta_norm = (fetta_norm - v_min) / (v_max - v_min + 1e-10)
        
        # 6. Orientamento Astronomico (Tipico FITS)
        fetta_norm = np.flipud(fetta_norm) 
        
        # 7. Applicazione Colormap (da float 0-1 a RGB uint8)
        rgba_frame = cmap(fetta_norm)
        rgb_frame = (rgba_frame[:, :, :3] * 255).astype(np.uint8)
        
        frames_rgb.append(rgb_frame)

    # 8. Salvataggio finale
    imageio.mimsave(output_path, frames_rgb, fps=fps, loop=0)
    
    piani = ["Spaziale (RA-Dec)", "RA-Frequenza", "Dec-Frequenza"]
    print(f"GIF Astro salvata: {output_path} (Piano: {piani[axis]})")
# Nel tuo script principale, aggiungi questo dopo la generazione di rec_np:

def run_comparative_plots(rec_np, output_dir, base_name="Ricostruzione_Astro"):
    # 1. Preprocessing 
    cube, vmin, vmax = preprocess(rec_np, use_log=False, use_percentile=True)
    
    print(f"\n--- Generazione Plot Comparativi per {base_name} ---")
    
    # 2. Griglia Statica (3x3 sezioni)
    static_grid(cube, vmin, vmax, base_name, output_dir=output_dir)
    
    # 3. Volume Rendering 3D (PyVista)
    # Nota: Assicurati che l'ambiente supporti il rendering grafico
    try:
        volume_rendering(cube, base_name, output_dir=output_dir)
        isosurface(cube, base_name, output_dir=output_dir)
    except Exception as e:
        print(f"Errore nel rendering 3D: {e}")

    # 4. Animazione (GIF)
    animate_slices(cube, vmin, vmax, base_name, output_dir=output_dir)


if __name__ == "__main__":
    flag = "ddim"
    if flag == "ddim":
        # ADATTARE PATH AL CASO ASTRO
        CHECKPOINT = Path("./data/outputs/BRGM_ddim_cond/checkpoint.pth") 
        RESULT_DIR = Path("./data/outputs/visualizzazione_ddim")
    elif flag == "decoder":
        CHECKPOINT = "C:\\Modelli 3D\\InverseSR - Astro\\outputs from Leonardo\\BRGM_decoder\\checkpoint.pth"
        RESULT_DIR = "./data/outputs/visualizzazione_decoder"

    DECODER_MODEL = "./data/trained_models_astro/decoder/data/model.pth"
   

    volume = visualize_reconstruction(CHECKPOINT, DECODER_MODEL, RESULT_DIR, flag=flag)
    run_comparative_plots(volume, output_dir=RESULT_DIR, base_name="Ricostruzione_Astro")
    # Prova a generare la vista ASSIALE (dall'alto verso il basso)
    
    """ for i in range(3):
        salva_datacube_gif(volume, f"{RESULT_DIR}/astro_axial_{i}.gif", axis=i, use_log=False) """
