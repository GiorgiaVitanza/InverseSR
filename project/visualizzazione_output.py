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
    # Questi sono valori tipici per l'architettura LDM Brain v2
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
        path_oggetto = Path(r"C:\Modelli 3D\InverseSR\data\trained_models\ddpm\data\model.pth")
        ddpm = torch.load(path_oggetto, weights_only=False, map_location="cpu")
        
        
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
    Adattato per Cubi FITS/Radioastronomia:
    axis=0: Scorre lungo la frequenza/velocità (Channel Map)
    axis=1: Scorre lungo la Declinazione (Piani RA-Freq)
    axis=2: Scorre lungo l'Ascensione Retta (Piani Dec-Freq)
    """
    # 1. Pulizia dimensioni extra (Squeeze) e gestione canali
    volume = np.squeeze(volume)
    if volume.ndim == 4:
        volume = volume[0] # Prendi il primo canale se presente

    # 2. Pre-processing per Astronomia (Log Scale)
    if use_log:
        # Evitiamo log(0) con un piccolo epsilon basato sulla dinamica dei dati
        eps = 1e-8
        volume = np.log10(volume + eps)

    # 3. Normalizzazione per GIF (0-255)
    v_min, v_max = volume.min(), volume.max()
    if v_max - v_min == 0:
        volume_norm = np.zeros_like(volume, dtype=np.uint8)
    else:
        volume_norm = ((volume - v_min) / (v_max - v_min) * 255).astype(np.uint8)
    
    # 4. Mappatura colori (opzionale ma consigliato)
    # imageio salva meglio se passiamo frame RGB. Usiamo 'inferno' di matplotlib.
    cm = plt.get_cmap('inferno')
    
    frames = []
    
    # 5. Generazione Frame
    for i in range(volume_norm.shape[axis]):
        if axis == 0:
            fetta = volume_norm[i, :, :] # Piano spaziale RA-Dec
        elif axis == 1:
            fetta = volume_norm[:, i, :] # Piano RA-Freq
        else:
            fetta = volume_norm[:, :, i] # Piano Dec-Freq
            
        # Squeeze finale di sicurezza per evitare l'errore Pillow (1, 1, 160)
        fetta = np.squeeze(fetta)
        
        # Ruotiamo e applichiamo l'origine 'lower' tipica dei FITS
        # origin='lower' in astro significa che la prima riga è in basso.
        fetta = np.flipud(fetta) 
        
        # Applichiamo la colormap per trasformare in RGB (0-255)
        fetta_rgb = (cm(fetta / 255.0)[:, :, :3] * 255).astype(np.uint8)
        
        frames.append(fetta_rgb)
    
    # 6. Salvataggio
    imageio.mimsave(output_path, frames, fps=fps, loop=0)
    
    # Etichette descrittive per il print
    piani = ["Spaziale (RA-Dec)", "RA-Frequenza", "Dec-Frequenza"]
    print(f"GIF Astro salvata: {output_path} (Piano: {piani[axis]})")

# Nel tuo script principale, aggiungi questo dopo la generazione di rec_np:

def run_comparative_plots(rec_np, base_name="Ricostruzione_Astro"):
    # 1. Preprocessing (molto importante per dati astronomici/logaritmici)
    # Usiamo la tua funzione preprocess definita all'inizio
    cube, vmin, vmax = preprocess(rec_np, use_log=True, use_percentile=True)
    
    print(f"\n--- Generazione Plot Comparativi per {base_name} ---")
    
    # 2. Griglia Statica (3x3 sezioni)
    static_grid(cube, vmin, vmax, base_name)
    
    # 3. Volume Rendering 3D (PyVista)
    # Nota: Assicurati che l'ambiente supporti il rendering grafico
    try:
        volume_rendering(cube, base_name)
        isosurface(cube, base_name)
    except Exception as e:
        print(f"Errore nel rendering 3D: {e}")

    # 4. Animazione (GIF)
    animate_slices(cube, vmin, vmax, base_name)


if __name__ == "__main__":
    flag = "decoder" # "ddim" o "decoder" 
    if flag == "ddim":
        # ADATTARE PATH AL CASO ASTRO
        CHECKPOINT = "C:\\Modelli 3D\\InverseSR\\codice GPU per Leonardo OK\\output_job_1_ddpm\\checkpoint.pth" 
        RESULT_DIR = "./data/outputs/visualizzazione_ddim"
    elif flag == "decoder":
        CHECKPOINT = "C:\\Modelli 3D\\InverseSR - Astro\\outputs from Leonardo\\outputs\\checkpoint.pth"
        RESULT_DIR = "./data/outputs/visualizzazione_decoder"

    DECODER_MODEL = "C:/Modelli 3D/InverseSR - Astro/data/trained_models_astro/decoder/data/model.pth"
   

    volume = visualize_reconstruction(CHECKPOINT, DECODER_MODEL, RESULT_DIR, flag=flag)
    run_comparative_plots(volume, base_name="Ricostruzione_Astro")
    # Prova a generare la vista ASSIALE (dall'alto verso il basso)
    for i in range(3):
        salva_datacube_gif(volume, f"{RESULT_DIR}/brain_axial_{i}.gif", axis=i)