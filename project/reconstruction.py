# 1. Definiamo le dimensioni
patch_shape = (32, 32, 32)
stride = (16, 16, 16) # Overlap del 50%
target_shape = (1, 128, 128, 128) # [Canali, Z, Y, X]

# 2. Generiamo coordinate e patch fittizi (simulando l'output del tuo modello)
patches = []
coords = []

for z in range(0, target_shape[1] - patch_shape[0] + 1, stride[0]):
    for y in range(0, target_shape[2] - patch_shape[1] + 1, stride[1]):
        for x in range(0, target_shape[3] - patch_shape[2] + 1, stride[2]):
            # Simuliamo un patch generato dal modello Super Resolution
            patch = torch.randn(1, *patch_shape) 
            
            patches.append(patch)
            coords.append((z, y, x))

patches = torch.stack(patches) # Tensor [N, C, pD, pH, pW]

# 3. Ricostruzione del cubo finale
cubo_ricostruito = reconstruct_volume_from_patches(
    patches=patches,
    coords=coords,
    original_shape=target_shape,
    stride=stride
)

print("Forma del cubo ricostruito:", cubo_ricostruito.shape)
# Output: torch.Size([1, 128, 128, 128])