import argparse
import torch

def get_config():
    parser = argparse.ArgumentParser(description="Unet Configuration - Deep 3D LDM Analysis")

    # --- Dimensioni canali ---
    # Per Cross-Attention su latenti VAE a 3 canali:
    # L'input z_t ha 3 canali e l'output della UNet (rumore/x0) ne ha 3.
    parser.add_argument("--in_channels_unet", type=int, default=3) 
    parser.add_argument("--out_channels_unet", type=int, default=3)
    
    # model_channels
    parser.add_argument("--model_channels", type=int, default=256) 
    
    # Risoluzione latente [D, H, W] oppure dimensione singola
    parser.add_argument(
        "--image_size", 
        type=int, 
        nargs='+', 
        default=[4, 32, 32], 
        help="Risoluzione latente"
    ) 
    
    # --- Parametri ResNet ---
    parser.add_argument("--num_res_blocks", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    
    # channel_mult: [1, 2] garantisce che la profondità D=4 passi a D=2 senza azzerarsi!
    parser.add_argument("--channel_mult", type=int, nargs='+', default=[1, 2]) 
    
    parser.add_argument("--conv_resample", action='store_true', default=True)
    parser.add_argument("--use_scale_shift_norm", action='store_true', default=True)
    parser.add_argument("--resblock_updown", action='store_true', default=True)

    # --- Parametri Attention & Transformer ---
    # Riduzioni spaziali alle quali applicare l'attenzione
    parser.add_argument("--attention_resolutions", type=int, nargs='+', default=[2, 1])
    
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_head_channels", type=int, default=64)

    # --- Condizionamento (Cross-Attention) ---
    # Per abilitare il meccanismo di attenzione sul contesto (es. i 4 parametri fisici)
    parser.add_argument("--use_spatial_transformer", action=argparse.BooleanOptionalAction,
        default=False,
        help="Abilita o disabilita il canale della maschera",)
    parser.add_argument(
        "--use_mask_channel",
        action=argparse.BooleanOptionalAction,  # oppure import argparse -> argparse.BooleanOptionalAction
        default=True,
        help="Abilita o disabilita il canale della maschera",
    )
    parser.add_argument("--transformer_depth", type=int, default=1)
    
    # Shape del context [B, 1, 4] o [B, 4] -> dimensione vettoriale = 4
    parser.add_argument("--context_dim", type=int, default=None) 
    
    args, unknown = parser.parse_known_args()

    return {"params": vars(args)}, unknown