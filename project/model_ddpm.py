import torch
# Assicurati che il percorso di import sia corretto per il tuo progetto
# from models.ddpm_v3d_conditioned import DDPM 

path_fisico = r"C:/Modelli 3D/InverseSR/data/trained_models/ddpm/data/model.pth"

try:
    # Carichiamo il checkpoint
    checkpoint = torch.load(path_fisico, weights_only=False, map_location="cpu")
    
    # Estraiamo lo state_dict
    state_dict = checkpoint.state_dict() if isinstance(checkpoint, torch.nn.Module) else checkpoint

    print("--- Analisi dei Canali dai Pesi (Versione 3D) ---")
    
    # 1. Trova in_channels (Conv3d usa 5 dimensioni)
    # Cerchiamo la prima chiave con 5 dimensioni: [Out, In, Depth, Height, Width]
    first_conv3d_key = next((k for k in state_dict.keys() if "weight" in k and state_dict[k].ndim == 5), None)
    
    if first_conv3d_key:
        shape = state_dict[first_conv3d_key].shape
        in_channels = shape[1] # La seconda dimensione è sempre in_channels
        print(f"Rilevata prima Conv3D: '{first_conv3d_key}'")
        print(f"Shape: [Out:{shape[0]}, In:{shape[1]}, D:{shape[2]}, H:{shape[3]}, W:{shape[4]}]")
        print(f"Inferred in_channels: {in_channels}")
    else:
        in_channels = "N/D (Nessuna Conv3D trovata)"

    # 2. Trova out_channels
    # Prendiamo l'ultima chiave con 5 dimensioni
    conv3d_keys = [k for k in state_dict.keys() if "weight" in k and state_dict[k].ndim == 5]
    last_conv3d_key = conv3d_keys[-1] if conv3d_keys else None

    if last_conv3d_key:
        shape = state_dict[last_conv3d_key].shape
        out_channels = shape[0] # La prima dimensione è out_channels
        print(f"Rilevata ultima Conv3D: '{last_conv3d_key}'")
        print(f"Inferred out_channels: {out_channels}")
    else:
        out_channels = "N/D"

    # --- Salvataggio Report ---
    file_path = "parametri_modello_ddpm_3d.txt"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("ANALISI MODELLO 3D (Conv3D detected)\n")
        f.write(f"In Channels (Input volumetrico): {in_channels}\n")
        f.write(f"Out Channels (Output volumetrico): {out_channels}\n")
        f.write("="*50 + "\n\n")
        
        for name, param in state_dict.items():
            s = list(param.shape)
            tipo_layer = "Conv3D/Pesi" if len(s) == 5 else "Bias/Altro"
            f.write(f"Layer: {name} | Tipo stimato: {tipo_layer} | Shape: {s}\n")
            f.write("-" * 30 + "\n")

    print(f"\nAnalisi 3D completata! Log: {file_path}")

except Exception as e:
    print(f"Errore durante l'analisi: {e}")