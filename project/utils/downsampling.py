import os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

# La tua classe di degrado
class ForwardDownsample:
    def __init__(self, factor, target_size=None):
        self.factor = factor
        self.target_size = target_size

    def __call__(self, x):        
        return F.interpolate(
            x,
            scale_factor=1 / self.factor,
            mode="trilinear",
            recompute_scale_factor=True,
            align_corners=False,
        )

def process_hr_to_lr(
    hr_dir: str | Path,
    lr_dir: str | Path,
    downsampler: ForwardDownsample,
    device: str = "cpu"
) -> None:
    """
    Legge tutti i patch .npy da hr_dir, applica ForwardDownsample e li salva in lr_dir.
    """
    hr_path = Path(hr_dir)
    lr_path = Path(lr_dir)
    lr_path.mkdir(parents=True, exist_ok=True)

    # Recupera tutti i file .npy
    npy_files = list(hr_path.glob("*.npy"))
    
    if not npy_files:
        print(f"Nessun file .npy trovato in: {hr_path}")
        return

    print(f"Trovati {len(npy_files)} file .npy. Inizio la conversione...")

    for file_path in npy_files:
        # 1. Caricamento del file .npy
        data_np = np.load(file_path)  # Shape attesa 3D (D, H, W) oppure 4D/5D

        # 2. Preparazione per PyTorch (F.interpolate richiede forma 5D per trilinear: [B, C, D, H, W])
        tensor_x = torch.from_numpy(data_np).float()

        # Aggiustamento delle dimensioni (Batch e Channel)
        original_ndim = tensor_x.ndim
        if original_ndim == 3:
            # Da (D, H, W) -> (1, 1, D, H, W)
            tensor_x = tensor_x.unsqueeze(0).unsqueeze(0)
        elif original_ndim == 4:
            # Da (C, D, H, W) -> (1, C, D, H, W)
            tensor_x = tensor_x.unsqueeze(0)
        elif original_ndim != 5:
            raise ValueError(f"Formato non supportato per {file_path.name}: {original_ndim}D")

        # Spostamento sul device (opzionale per accelerare su GPU)
        tensor_x = tensor_x.to(device)

        # 3. Applicazione della funzione di downsampling
        with torch.no_grad():
            tensor_lr = downsampler(tensor_x)

        # 4. Ripristino della forma originale (Rimuove le dimensioni Batch/Channel aggiunte)
        tensor_lr = tensor_lr.cpu()
        if original_ndim == 3:
            array_lr = tensor_lr.squeeze(0).squeeze(0).numpy()
        elif original_ndim == 4:
            array_lr = tensor_lr.squeeze(0).numpy()
        else:
            array_lr = tensor_lr.numpy()

        # 5. Salvataggio nella cartella LR
        output_file = lr_path / file_path.name
        np.save(output_file, array_lr)

    print(f"Completato! File salvati in: {lr_path}")


# --- Esempio di utilizzo ---
if __name__ == "__main__":
    # Configurazione
    HR_FOLDER = "../../data/inputs/16x128x128_cont_ldev_OK/train_augmented"
    LR_FOLDER = "../../data/inputs/16x128x128_cont_ldev_OK/train_augmented_LR"
    DOWNSAMPLE_FACTOR = 4

    # Inizializzazione della classe
    downsampler = ForwardDownsample(factor=DOWNSAMPLE_FACTOR)

    # Esecuzione
    process_hr_to_lr(
        hr_dir=HR_FOLDER,
        lr_dir=LR_FOLDER,
        downsampler=downsampler,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )