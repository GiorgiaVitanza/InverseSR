import argparse
import torch

def get_hparams():
    parser = argparse.ArgumentParser(description="Training Vae Configuration - 3D Astro Analysis")

    # --- Percorsi File e Logging ---
    parser.add_argument("--data_dir", type=str, default="./data/inputs/patches_160_224_160_stride160", help="Path alla cartella dei dati")
    parser.add_argument("--catalogue_path", type=str, default="./data/inputs/sky_dev_truthcat_v2.txt", help="Path al file txt del catalogo")
    parser.add_argument("--tensor_board_logger", type=str, default="logs/", help="Path per il logger di TensorBoard")
    parser.add_argument("--output_dir", type=str, default="outputs/", help="Cartella per salvare i grafici e i pesi")

    # --- Parametri Architettura (Adattati all'analisi Conv3D) ---
    parser.add_argument("--in_channels", type=int, default=1, help="Canali input (Input volumetrico: 1)")
    
    # Impostato a 3 come richiesto nei tuoi parametri iniziali
    parser.add_argument("--out_channels", type=int, default=3, help="Canali output (Output volumetrico: 3)")
    
    # z_channels rilevato dai layer quant_conv_mu
    parser.add_argument("--z_channels", type=int, default=3, help="Canali latenti stimati (Z-Channels)")
    
    # n_channels rilevato dal primo blocco dell'encoder [64, 1, 3, 3, 3]
    parser.add_argument("--n_channels", type=int, default=64, help="Numero di canali base (rilevato: 64)")
    
    # Basato sulla sequenza di layer encoder.blocks.1, 2, 4, 5 etc.
    parser.add_argument("--num_res_blocks", type=int, default=2, help="Numero di blocchi residui per livello")
    
    # ch_mult: downsample 1x2x2 = 4
    parser.add_argument("--ch_mult", type=int, nargs='+', default=[1, 2, 2], help="Moltiplicatori canali (64 -> 128)")
    
    parser.add_argument("--resolution", type=int, nargs='+', default=[160, 224, 160], help="Risoluzione spaziale del datacube")
    parser.add_argument("--attn_resolutions", type=int, nargs='+', default=[], help="Risoluzioni per Self-Attention (non rilevata nell'analisi)")

    # --- Parametri Training ---
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (consigliato 1 per cubi 128^3 e GPU standard)")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--experiment_name", type=str, default="Astro_VAE_Conv3D_Deep")
    
    
    # Device Management
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()
