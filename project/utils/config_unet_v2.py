import argparse
import torch

def get_config():
    parser = argparse.ArgumentParser(description="Unet Configuration")

    # --- Dimensioni canali ---
    # Fondamentale: l'input della UNet è l'output dell'encoder (z_channels=3)
    parser.add_argument("--in_channels", type=int, default=3) 
    # L'output deve corrispondere all'input per poter calcolare la loss sul rumore (z_channels=3)
    parser.add_argument("--out_channels", type=int, default=3)
    
    parser.add_argument("--model_channels", type=int, default=128) # Aumentato per gestire la complessità
    
    # Fondamentale: la dimensione su cui lavora la UNet è quella latente (32)
    parser.add_argument("--image_size", type=int, default=32) 
    
    # --- Parametri ResNet ---
    parser.add_argument("--num_res_blocks", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    
    # ch_mult: con image_size=32, una lista [1, 2, 4] porterà la risoluzione interna a 32->16->8
    # Evita di scendere sotto 8x8x8 nei cubi astrofisici per non perdere troppa coerenza spaziale
    parser.add_argument("--channel_mult", type=int, nargs='+', default=[1, 2, 4]) 
    
    parser.add_argument("--conv_resample", action='store_true', default=True)
    parser.add_argument("--use_scale_shift_norm", action='store_true', default=True)
    parser.add_argument("--resblock_updown", action='store_true', default=True)

    # --- Parametri Attention ---
    # L'attenzione si attiva quando la risoluzione è bassa (es. 16x16x16 e 8x8x8)
    parser.add_argument("--attention_resolutions", type=int, nargs='+', default=[16, 8])
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_head_channels", type=int, default=32)
    parser.add_argument("--num_heads_upsample", type=int, default=-1)

    # --- Condizionamento e Transformer (LDM Style) ---
    parser.add_argument("--use_spatial_transformer", action="store_true", default=True)
    parser.add_argument("--transformer_depth", type=int, default=1)
    
    # context_dim: se condizioni con un catalogo (es. coordinate, magnitudine), 
    # specifica qui la dimensione del vettore di embedding del catalogo.
    parser.add_argument("--context_dim", type=int, default=128) 

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