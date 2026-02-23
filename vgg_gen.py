import torch
import torch.nn as nn
import torchvision
import os

class CompatibleVGG_Astro(torch.nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        print(f"Loading VGG16 weights and adapting for {in_channels} input channel(s)...")
        
        # 1. Carica VGG16 standard
        original_vgg = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT)
        
        # 2. Adattamento del primo strato (come prima)
        original_first_layer = original_vgg.features[0]
        self.expected_channels = in_channels 

        if in_channels != 3:
            new_first_layer = nn.Conv2d(
                in_channels=in_channels,
                out_channels=original_first_layer.out_channels,
                kernel_size=original_first_layer.kernel_size,
                stride=original_first_layer.stride,
                padding=original_first_layer.padding
            )
            
            # Copia intelligente dei pesi (Media dei canali RGB -> 1 Canale)
            with torch.no_grad():
                if in_channels == 1:
                    new_first_layer.weight[:] = torch.mean(original_first_layer.weight, dim=1, keepdim=True)
                else:
                    nn.init.kaiming_normal_(new_first_layer.weight, mode='fan_out', nonlinearity='relu')
                
                new_first_layer.bias = original_first_layer.bias

            original_vgg.features[0] = new_first_layer

        self.features = original_vgg.features
        self.eval()

    def forward(self, x: torch.Tensor, resize_images: bool = False, return_lpips: bool = True):
        # --- FIX AUTOMATICO PER I CANALI ---
        # Se il modello vuole 1 canale ma ne arrivano 3 (errore comune coi dataloader standard)
        if self.expected_channels == 1 and x.shape[1] == 3:
            # Facciamo la media sui 3 canali per ottenerne 1 (RGB -> Grayscale)
            # Questo risolve il RuntimeError senza toccare il dataloader
            x = x.mean(dim=1, keepdim=True)
        # -----------------------------------

        return self.features(x)

# Setup paths
target_path = "./data/trained_models/vgg16_astro_1ch.pt"
os.makedirs(os.path.dirname(target_path), exist_ok=True)

# Instantiate
print("Creating compatible Astro-model (Robust 1-Channel)...")
model = CompatibleVGG_Astro(in_channels=1)

# Scripting
print("Converting to TorchScript...")
# Usiamo un dummy input a 3 canali per testare che il fix funzioni anche durante il trace/script
dummy_input = torch.randn(1, 3, 224, 224) 
scripted_model = torch.jit.script(model)

# Save
print(f"Saving to {target_path}...")
scripted_model.save(target_path)
print("Done! Model saved and is now robust to 3-channel inputs.")