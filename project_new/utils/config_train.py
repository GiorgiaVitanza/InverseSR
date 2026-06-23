import argparse
import torch

def train_config():

    parser = argparse.ArgumentParser(description="Training DDPM Configuration")



    # --- Paths ---

    parser.add_argument("--data_dir", type=str, default="./data/inputs/128x128x128_stride128/train/npy_patches", help="Path alla cartella dei dati")
    
    parser.add_argument("--test_dir", type=str, default="./data/inputs/128x128x128_stride128/test/npy_patches", help="Path alla cartella dei dati di test")

    parser.add_argument("--test_fig", type=str, default="./data/outputs/test_ddpm", help="Path alla cartella dei plot di test")
    
    parser.add_argument("--vae_path", type=str, default="/leonardo_scratch/large/userexternal/gvitanza/InverseSR/checkpoints_vae_decoder_8_10epochs_May04_19-02-45/vae_full_ep10.pth", help="Path alla cartella dei pesi del VAE")

    parser.add_argument("--output_dir_vae", type=str, default="./data/outputs/vae", help="Path alla cartella di output")

    parser.add_argument("--output_dir_ddpm", type=str, default="./data/outputs/ddpm", help="Path alla cartella di output")

    parser.add_argument("--catalogue_path", type=str, default="./data/inputs/128x128x128/test/test_catalog.csv", help="Path al file txt del catalogo")

    parser.add_argument("--tensor_board_logger_vae", type=str, default="./logs_vae", help="Path per il logger di TensorBoard")

    parser.add_argument("--tensor_board_logger_ddpm", type=str, default="./logs_ddpm", help="Path per il logger di TensorBoard")
    
    # Conditioning per la unet 
    parser.add_argument("--cond_key", type=str, default=None)

    # --- Parametri Training ---
    parser.add_argument("--norm_mode", type=str, default="global_sym", help="Modalità di normalizzazione dei dati")
    
    parser.add_argument("--batch_size", type=int, default=1)

    parser.add_argument("--learning_rate", type=float, default=1e-4)

    parser.add_argument("--epochs", type=int, default=10)

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")



    args, unknown = parser.parse_known_args()
    return args, unknown
