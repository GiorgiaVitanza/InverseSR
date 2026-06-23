import argparse
import torch

def get_hparams():
    parser = argparse.ArgumentParser(description="Vae Configuration - 3D Astro Analysis")


    # --- Parametri Architettura (Adattati all'analisi Conv3D) ---
    parser.add_argument("--in_channels", type=int, default=1, help="Canali input (Input volumetrico: 1)")
    parser.add_argument("--out_channels", type=int, default=1, help="Canali output (Output volumetrico: 1)")
    
    # z_channels rilevato dai layer quant_conv_mu
    parser.add_argument("--z_channels", type=int, default=16, help="Canali latenti stimati (Z-Channels)")
    
    # n_channels rilevato dal primo blocco dell'encoder [64, 1, 3, 3, 3]
    parser.add_argument("--n_channels", type=int, default=64, help="Numero di canali base (rilevato: 64)")
    
    # Basato sulla sequenza di layer encoder.blocks.1, 2, 4, 5 etc.
    parser.add_argument("--num_res_blocks", type=int, default=2, help="Numero di blocchi residui per livello")
    
    # ch_mult: downsample 1x2x2 = 4
    parser.add_argument("--ch_mult", type=int, nargs='+', default=[1, 2, 2], help="Moltiplicatori canali (64 -> 128)")
    
    parser.add_argument("--resolution", type=int, nargs='+', default=[160, 224, 160], help="Risoluzione spaziale del datacube")
    parser.add_argument("--attn_resolutions", type=int, nargs='+', default=[], help="Risoluzioni per Self-Attention (non rilevata nell'analisi)")

    
    args, unknown = parser.parse_known_args(); 
    
    return args, unknown
    
