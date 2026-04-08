import torch
import torch.nn as nn
import torchvision
import os

class AstroVGG_Slim(nn.Module):
    """
    Versione ottimizzata di VGG16 per dati astrofisici (1 canale).
    - Adattata per input monocromatico (in_channels=1).
    - Ridotto il numero di pooling per gestire patch piccole.
    - Inplace=False per compatibilità gradient checkpointing.
    """
    def __init__(self, target_path=None, in_channels=1, num_blocks=2):
        super().__init__()
        print(f"Inizializzazione AstroVGG Slim ({num_blocks} blocchi, {in_channels} ch)...")
        
        # 1. Carica l'architettura VGG16
        vgg = torchvision.models.vgg16(weights=None)
        
        # 2. ADATTAMENTO PER 1 CANALE (Modifica del primo layer)
        # Il layer originale è: Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        if in_channels != 3:
            old_layer = vgg.features[0]
            vgg.features[0] = nn.Conv2d(
                in_channels, 
                old_layer.out_channels, 
                kernel_size=old_layer.kernel_size, 
                stride=old_layer.stride, 
                padding=old_layer.padding
            )

        # 3. Taglio della rete (Truncation)
        if num_blocks == 1:
            cut_off = 5
        elif num_blocks == 2:
            cut_off = 10
        else:
            cut_off = 17 
            
        self.features = nn.Sequential(*list(vgg.features)[:cut_off])

        # 4. Caricamento pesi (se il file esiste)
        if target_path and os.path.exists(target_path):
            print(f"Caricamento pesi AstroVGG da: {target_path}")
            # Usiamo strict=False se stiamo caricando pesi da una VGG a 3 canali su una a 1,
            # altrimenti strict=True se il file .pth è già della versione a 1 canale.
            try:
                self.load_state_dict(torch.load(target_path), strict=True, weights_only=False)
            except RuntimeError:
                print("Nota: Caricamento parziale dei pesi (probabile mismatch canali input).")
                self.load_state_dict(torch.load(target_path), strict=False, weights_only=False)

        # 5. DISABILITA INPLACE
        for m in self.features.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = False
        
        self.eval()

    def forward(self, x: torch.Tensor):
        # Protezione dimensionale: VGG soffre con input < 32x32 se ci sono troppi pool
        if x.shape[2] < 16 or x.shape[3] < 16:
            x = torch.nn.functional.interpolate(x, size=(32, 32), mode='bilinear', align_corners=False)
            
        return self.features(x)

if __name__ == "__main__":
    # --- CONFIGURAZIONE ---
    target_dir = "./data/trained_models_astro/vgg"
    target_path = os.path.join(target_dir, "vgg16_slim_astro_1ch.pth")
    os.makedirs(target_dir, exist_ok=True)

    # 1. Crea il modello per 1 CANALE
    model = AstroVGG_Slim(in_channels=1, num_blocks=2)

    # 2. Test rapido
    # Dummy input ora ha 1 solo canale (Batch, Channel, H, W)
    dummy_input = torch.randn(1, 1, 28, 28) 
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"\nTest Success!")
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    # 3. Salvataggio
    print(f"Salvataggio pesi in: {target_path}...")
    torch.save(model.state_dict(), target_path)
    
    print("\n--- SCRIPT COMPLETATO CON SUCCESSO ---")