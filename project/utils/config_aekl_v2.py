import argparse
import torch

def get_hparams():
    parser = argparse.ArgumentParser(description="Training Vae Configuration")

    # --- Percorsi File e Logging ---
    parser.add_argument("--data_dir", type=str, default="./data/inputs/patches", help="Path alla cartella dei dati")

    parser.add_argument("--catalogue_path", type=str, default="./data/inputs/sky_dev_truthcat_v2.txt", help="Path al file txt del catalogo")
    parser.add_argument("--tensor_board_logger", type=str, default="logs/", help="Path per il logger di TensorBoard")
    parser.add_argument("--output_dir", type=str, default="outputs/", help="Cartella per salvare i grafici e i pesi")

    # --- Parametri Architettura (Coerenti con LATENT_SHAPE [1, 3, 32, 32, 32]) ---
    parser.add_argument("--in_channels", type=int, default=1, help="Canali input (solitamente 1 per FITS)")
    parser.add_argument("--out_channels", type=int, default=1, help="Canali output")
    
    # z_channels deve essere 3 per corrispondere al tuo LATENT_SHAPE
    parser.add_argument("--z_channels", type=int, default=3, help="Dimensione dei canali nello spazio latente")
    
    # n_channels controlla la complessità degli strati intermedi (es. 32 o 64)
    parser.add_argument("--n_channels", type=int, default=32, help="Numero di canali base nei layer convoluzionali")
    
    parser.add_argument("--num_res_blocks", type=int, default=2, help="Numero di blocchi residui per ogni livello")
    
    # ch_mult: con [1, 2] ottieni un downsampling di 4x (2x per ogni livello)
    # Se volessi 8x, useresti [1, 2, 4]
    parser.add_argument("--ch_mult", type=int, nargs='+', default=[1, 2], help="Moltiplicatore canali per livello (lunghezza = num_downsamplings)")
    
    parser.add_argument("--resolution", type=int, nargs='+', default=[128, 128, 128], help="Risoluzione spaziale del datacube")
    parser.add_argument("--attn_resolutions", type=int, nargs='+', default=[], help="Risoluzioni a cui applicare Self-Attention")

    # --- Parametri Training ---
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (consigliato 1-2 per cubi 128^3)")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--experiment_name", type=str, default="Astro_VAE_Exp")
    
    # Parametri per Inverse Problems (es. Super Resolution)
    parser.add_argument("--lambda_perc", type=float, default=0.01, help="Peso della Perceptual Loss (se usata)")
    parser.add_argument("--slicing_dim", type=str, default="spatial", choices=["spatial", "spectral"], help="Dimensione su cui operare lo slicing")
    parser.add_argument("--corruption", type=str, default="none", help="Tipo di degradazione per testare l'inverse problem")
    
    # Device Management
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()