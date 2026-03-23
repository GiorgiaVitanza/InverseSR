import argparse
import torch

def train_config():

    parser = argparse.ArgumentParser(description="Training DDPM Configuration")



    # --- Paths ---

    parser.add_argument("--data_dir", type=str, default="./data/inputs/patches_160_224_160_stride160", help="Path alla cartella dei dati")

    parser.add_argument("--vae_path", type=str, default="./data/trained_models_astro/checkpoints_vae_decoder/vae_full_ep100.pth", help="Path alla cartella dei pesi del VAE")

    parser.add_argument("--output_dir", type=str, default="./data/outputs/ddpm", help="Path alla cartella di output")

    parser.add_argument("--catalogue_path", type=str, default="./data/inputs/sky_dev_truthcat_v2.txt", help="Path al file txt del catalogo")

    parser.add_argument("--tensor_board_logger", type=str, default="./data/outputs/logs_ddpm", help="Path per il logger di TensorBoard")
    
    # Conditioning per la unet 
    parser.add_argument("--cond_key", type=str, default=None)

    # --- Parametri Training ---

    parser.add_argument("--batch_size", type=int, default=1)

    parser.add_argument("--learning_rate", type=float, default=1e-4)

    parser.add_argument("--epochs", type=int, default=10)

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")



    args, unknown = parser.parse_known_args()
    return args, unknown
