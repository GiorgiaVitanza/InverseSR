import numpy as np
from astropy.io import fits
import os
from tqdm import tqdm

def process_radio_cube(fits_path, output_dir, patch_size=(160, 224, 160), stride=160, filter_empty=True, threshold=0.0):
    """
    Taglia un cubo FITS in patch 5D (1, 1, D, H, W).
    
    Args:
        fits_path (str): Percorso del file .fits
        output_dir (str): Cartella dove salvare i patch .npy
        patch_size (tuple): Dimensioni del ritaglio (Z, Y, X) -> (Profondità, Altezza, Larghezza)
        stride (int): Passo di scorrimento. Se stride < patch_size, c'è sovrapposizione (utile per data augmentation).
        filter_empty (bool): Se True, scarta i patch che contengono solo rumore (sotto la soglia).
        threshold (float): Soglia di segnale minimo per considerare il patch valido.
    """
    
    # 1. Caricamento del FITS
    print(f"Caricamento file: {fits_path}")
    with fits.open(fits_path, memmap=True) as hdul:
        # Assumiamo che i dati siano nel primary HDU.
        # I cubi radio sono spesso (Stokes, Freq, Dec, RA) o (Freq, Dec, RA)
        data = hdul[0].data
        
    # 2. Pulizia e Gestione Dimensioni
    # Rimuoviamo assi degeneri (es. Stokes I se è dimensione 1)
    data = np.squeeze(data)
    
    # Sostituzione dei NaNs con 0 o con il valore minimo (importante per le CNN)
    if np.isnan(data).any():
        print("Rimozione NaNs...")
        data = np.nan_to_num(data, nan=0.0)

    # Verifica che il cubo sia 3D (Z, Y, X) dopo lo squeeze
    if data.ndim != 3:
        raise ValueError(f"Il dato deve essere 3D dopo lo squeeze. Dimensioni attuali: {data.shape}")

    print(f"Dimensioni Cubo Originale: {data.shape}")
    
    # Dimensioni target
    d_z, d_y, d_x = patch_size
    # Dimensioni attuali
    Z, Y, X = data.shape

    # Verifica che il patch non sia più grande del cubo
    if Z < d_z or Y < d_y or X < d_x:
        raise ValueError("Le dimensioni del patch sono più grandi del cubo stesso!")

    # Creazione cartella output
    os.makedirs(output_dir, exist_ok=True)

    patch_count = 0
    
    # 3. Ciclo di Ritaglio (Sliding Window)
    # Iteriamo su Z (Frequenza/Velocità), Y (Dec), X (RA)
    print("Inizio estrazione patch...")
    
    # tqdm per barra di avanzamento
    for z in tqdm(range(0, Z - d_z + 1, stride), desc="Scansione Spettrale"):
        for y in range(0, Y - d_y + 1, stride):
            for x in range(0, X - d_x + 1, stride):
                
                # Estrazione del cubo
                patch = data[z:z+d_z, y:y+d_y, x:x+d_x]
                
                # 4. Filtraggio (Opzionale)
                # Se il picco massimo nel patch è rumore, lo saltiamo
                if filter_empty and np.max(patch) <= threshold:
                    continue

                # 5. Reshape per formato ML (1, 1, 128, 128, 128)
                # Formato tipico PyTorch: (Batch, Channel, Depth, Height, Width)
                # Qui aggiungiamo Batch=1 e Channel=1
                patch_reshaped = patch[np.newaxis, np.newaxis, :, :, :]
                
                # Verifica finale shape
                expected_shape = (1, 1, d_z, d_y, d_x)
                assert patch_reshaped.shape == expected_shape, f"Shape errata: {patch_reshaped.shape}"
                
                # 6. Salvataggio
                filename = os.path.join(output_dir, f"patch_{patch_count:06d}.npy")
                np.save(filename, patch_reshaped.astype(np.float32))
                patch_count += 1

    print(f"--- Finito ---")
    print(f"Totale patch salvati: {patch_count}")
    print(f"Salvati in: {output_dir}")

# --- ESEMPIO DI UTILIZZO ---
if __name__ == "__main__":
    # Sostituisci con il tuo file reale
    INPUT_FITS = "./data/inputs/sky_dev_v2.fits" 
    OUTPUT_FOLDER = "./data/inputs/patches_160_224_160_stride160"
    
    # Configurazione
    # Nota: Stride 64 con dimensione 128 significa 50% di sovrapposizione (ottimo per training)
    # Se vuoi patch unici senza sovrapposizione, metti stride=128
    
    try:
        process_radio_cube(
            fits_path=INPUT_FITS,
            output_dir=OUTPUT_FOLDER,
            patch_size=(160, 224, 160),
            stride=160,           # Sovrapposizione per aumentare i dati
            filter_empty=True,   # Evita di salvare cielo vuoto
            threshold=0.001      # Imposta in base all'unità del tuo FITS (es. Jy/beam)
        )
    except FileNotFoundError:
        print("Errore: File FITS non trovato. Modifica la variabile INPUT_FITS.")
    except Exception as e:
        print(f"Errore generico: {e}")