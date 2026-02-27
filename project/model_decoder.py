import torch
import os

# Percorso del tuo decoder
path_fisico = r"C:/Modelli 3D/InverseSR/data/trained_models/decoder/data/model.pth"
file_report = "report_parametri_decoder_3d.txt"

try:
    # 1. Caricamento Checkpoint
    checkpoint = torch.load(path_fisico, weights_only=False, map_location="cpu")
    
    # Estrazione state_dict (gestisce sia modelli interi che dizionari)
    state_dict = checkpoint.state_dict() if hasattr(checkpoint, 'state_dict') else checkpoint

    print("--- Analisi Architettura Decoder 3D ---")

    # 2. Identificazione parametri chiave
    # Cerchiamo pesi con 5 dimensioni: [Out, In, D, H, W]
    layers_3d = [k for k in state_dict.keys() if "weight" in k and state_dict[k].ndim == 5]

    if not layers_3d:
        raise ValueError("Nessun layer Conv3D o ConvTranspose3D trovato nel file.")

    # --- INPUT CHANNELS (Latent Space) ---
    # Di solito è la seconda dimensione del PRIMO layer 3D
    first_key = layers_3d[0]
    shape_in = state_dict[first_key].shape
    in_channels = shape_in[1] 

    # --- OUTPUT CHANNELS (Ricostruzione) ---
    # Di solito è la prima dimensione dell'ULTIMO layer 3D
    last_key = layers_3d[-1]
    shape_out = state_dict[last_key].shape
    out_channels = shape_out[0]

    print(f"Primo Layer rilevato: '{first_key}' -> In Channels: {in_channels}")
    print(f"Ultimo Layer rilevato: '{last_key}' -> Out Channels: {out_channels}")

    # --- Generazione Report TXT ---
    with open(file_report, "w", encoding="utf-8") as f:
        f.write("RELAZIONE TECNICA: PARAMETRI DECODER 3D\n")
        f.write("="*50 + "\n")
        f.write(f"File sorgente: {os.path.basename(path_fisico)}\n")
        f.write(f"Canali Input (Latent): {in_channels}\n")
        f.write(f"Canali Output (3D Voxels): {out_channels}\n")
        f.write(f"Numero totale layer analizzati: {len(state_dict)}\n")
        f.write("="*50 + "\n\n")

        f.write(f"{'Nome Parametro':<60} | {'Tipo':<15} | {'Shape (O, I, D, H, W)'}\n")
        f.write("-" * 110 + "\n")

        for name, param in state_dict.items():
            shape = list(param.shape)
            
            # Etichettatura intelligente
            if len(shape) == 5:
                tipo = "VOLUMETRIC"
            elif len(shape) == 1:
                tipo = "BIAS/NORM"
            else:
                tipo = "OTHER"
            
            f.write(f"{name:<60} | {tipo:<15} | {str(shape)}\n")

    print(f"\nReport salvato con successo: {file_report}")

except Exception as e:
    print(f"Errore durante l'estrazione: {e}")