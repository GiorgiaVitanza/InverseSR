import torch

path_fisico = r"C:/Modelli 3D/InverseSR/data/trained_models/decoder/data/model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


try:
    # Carichiamo l'intero oggetto. 
    # weights_only=False è necessario se il file è un modello serializzato interamente.
    decoder_model = torch.load(path_fisico, map_location=device, weights_only=False)
    # Stampa il nome della classe
    print(f"La classe del modello è: {decoder_model.__class__.__name__}")
    print(f"Il modulo di origine è: {decoder_model.__class__.__module__}")
    # Se il file salvato era un dizionario {'state_dict': ...}, estraiamo solo i pesi
    if isinstance(decoder_model, dict) and 'state_dict' in decoder_model:
        # Nota: Qui serve che l'istanza della classe VAE sia già creata (es. vae_anon = VAE())
        # vae_anon.load_state_dict(vae_model['state_dict'])
        # vae_model = vae_anon
        print("Il file contiene uno state_dict. Assicurati di aver inizializzato la classe prima.")
    
    decoder_model.eval() # Fondamentale per l'inferenza
    print(f"Modello caricato con successo su {device}")

except Exception as e:
    print(f"Errore fatale nel caricamento: {e}")