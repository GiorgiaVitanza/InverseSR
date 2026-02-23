import argparse
import torch

def get_config():
    parser = argparse.ArgumentParser(description="Unet Configuration - Deep 3D LDM Analysis")

    # --- Dimensioni canali ---
    # Rilevato da input_blocks.0.0.weight: Shape [256, 7, 3, 3, 3]
    # L'input ha 7 canali (probabilmente z_latenti + condizionamento spaziale)
    parser.add_argument("--in_channels", type=int, default=7) 
    
    # Rilevato dai parametri iniziali dell'analisi
    parser.add_argument("--out_channels", type=int, default=3)
    
    # model_channels rilevato: 256 (il primo blocco proietta a 256)
    parser.add_argument("--model_channels", type=int, default=256) 
    
    # image_size: manteniamo 32 per coerenza con lo spazio latente del VAE
    parser.add_argument("--image_size", type=int, default=32) 
    
    # --- Parametri ResNet ---
    # Dalla numerazione dei blocchi (es. input_blocks.1, 2, 3 hanno tutti 256 canali)
    # si deduce una profondità di 3 blocchi per risoluzione
    parser.add_argument("--num_res_blocks", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0) # Spesso 0 in fase di inference/analisi pesi
    
    # channel_mult: 
    # blocks 0-3: 256 canali (mult 1)
    # blocks 4-6: 512 canali (mult 2)
    # blocks 7-8: 768 canali (mult 3)
    parser.add_argument("--channel_mult", type=int, nargs='+', default=[1, 2, 3]) 
    
    parser.add_argument("--conv_resample", action='store_true', default=True)
    parser.add_argument("--use_scale_shift_norm", action='store_true', default=True)
    parser.add_argument("--resblock_updown", action='store_true', default=True)

    # --- Parametri Attention & Transformer ---
    # L'attenzione appare dal blocco 4.1 (quando i canali diventano 512)
    # In una UNet con 3 livelli [32, 16, 8], il blocco 4 corrisponde solitamente alla risoluzione 16 o 8
    parser.add_argument("--attention_resolutions", type=int, nargs='+', default=[16, 8])
    
    # num_head_channels: calcolato dai pesi dell'attenzione [512, 512] 
    # Se num_heads è 4, allora head_dim è 16.
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_head_channels", type=int, default=16)

    # --- Condizionamento (Cross-Attention) ---
    parser.add_argument("--use_spatial_transformer", action="store_true", default=True)
    parser.add_argument("--transformer_depth", type=int, default=1)
    
    # context_dim rilevato: 4!!
    # Guarda layer: attn2.to_k.weight | Shape: [512, 4]
    # Questo indica che il modello viene condizionato da un vettore molto piccolo (es. 4 parametri fisici o di catalogo)
    parser.add_argument("--context_dim", type=int, default=4) 

    return {"params": vars(parser.parse_args())}

def train_config():

    parser = argparse.ArgumentParser(description="Training DDPM Configuration")



    # --- Paths ---

    parser.add_argument("--data_dir", type=str, default="./data/inputs/patches", help="Path alla cartella dei dati")

    parser.add_argument("--catalogue_path", type=str, default="./data/inputs/sky_dev_truthcat_v2.txt", help="Path al file txt del catalogo")

    parser.add_argument("--tensor_board_logger", type=str, default="./data/outputs/logs_ddpm", help="Path per il logger di TensorBoard")

   

    # --- Parametri Training ---

    parser.add_argument("--batch_size", type=int, default=1)

    parser.add_argument("--learning_rate", type=float, default=1e-4)

    parser.add_argument("--epochs", type=int, default=100)

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")



    # Restituisce un oggetto Namespace (accedi con args.param)

    return parser.parse_args()