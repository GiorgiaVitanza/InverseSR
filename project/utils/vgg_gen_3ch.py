import torch
import torch.nn as nn
import torchvision
import os

class AstroVGG_Slim(nn.Module):
    """
    Versione ottimizzata di VGG16 per dati astrofisici.
    - Ridotto il numero di pooling (2 invece di 5) per gestire patch piccole.
    - Disabilitato 'inplace' per compatibilità con Gradient Checkpointing.
    - Adattato per input a 3 canali (latente).
    """
    def __init__(self, target_path, in_channels=3, num_blocks=2):
        super().__init__()
        print(f"Inizializzazione AstroVGG Slim ({num_blocks} blocchi, {in_channels} ch)...")
        
        # 1. Carica VGG16 standard con pesi pre-addestrati
        vgg = torchvision.models.vgg16(weights=None)
        

        # 3. Taglio della rete (Truncation)
        # Block 1: layers 0-4 (MaxPool a indice 4)
        # Block 2: layers 5-9 (MaxPool a indice 9)
        # Fermandoci a 10, abbiamo 2 pooling totali (riduzione risoluzione 4x)
        if num_blocks == 1:
            cut_off = 5
        elif num_blocks == 2:
            cut_off = 10
        else:
            cut_off = 17 # 3 blocchi (riduzione 8x)
            
        self.features = nn.Sequential(*list(vgg.features)[:cut_off])
        if target_path and os.path.exists(target_path):
            print(f"Caricamento pesi AstroVGG da: {target_path}")
            # strict=True ora funzionerà perché abbiamo preparato l'architettura
            self.load_state_dict(torch.load(target_path), strict=True)

        # 4. DISABILITA INPLACE (Fondamentale per evitare CUDA Illegal Memory Access)
        for m in self.features.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = False
        
        self.eval() # Modalità inferenza (non addestriamo la VGG)

    def forward(self, x: torch.Tensor):
        # Protezione dimensionale: se l'input è troppo piccolo per i pooling, upsample
        # Con 2 blocchi, serve un minimo di 4x4 pixel. Usiamo 32 per sicurezza.
        if x.shape[2] < 16 or x.shape[3] < 16:
            x = torch.nn.functional.interpolate(x, size=(32, 32), mode='bilinear', align_corners=False)
            
        return self.features(x)

if __name__ == "__main__":
    # --- CONFIGURAZIONE SALVATAGGIO ---
    target_dir = "././data/trained_models_astro/vgg"
    target_path = os.path.join(target_dir, "vgg16_slim_astro.pth")
    os.makedirs(target_dir, exist_ok=True)

    # 1. Crea il modello
    # Usiamo 2 blocchi: ottimo compromesso tra memoria e dettaglio morfologico
    model = AstroVGG_Slim(target_path, in_channels=1, num_blocks=2)

    # 2. Test rapido di funzionamento
    print("Esecuzione test dimensionale...")
    dummy_input = torch.randn(1, 3, 28, 28) # Test con patch piccola (es. da spazio latente)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Test Success! Input: {dummy_input.shape} -> Output: {output.shape}")

    # 3. Salvataggio Pesi (State Dict)
    # NOTA: Evitiamo TorchScript/JIT per massima compatibilità con l'inversione BRGM
    print(f"Salvataggio pesi in: {target_path}...")
    torch.save(model.state_dict(), target_path)
    
    print("\n--- SCRIPT COMPLETATO CON SUCCESSO ---")
    print("Ora puoi caricare questo modello nel tuo script principale usando:")
    print("model.load_state_dict(torch.load('path_to_filei'))")
