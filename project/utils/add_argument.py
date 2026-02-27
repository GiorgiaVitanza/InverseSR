from argparse import ArgumentParser

import torch

def add_argument(parser: ArgumentParser):
    # --- LOGGING & PATHS ---
    parser.add_argument(
        "--tensor_board_logger",
        # Modificato: Path generico per astro
        default=r"./logs/astro_diffusion",
        help="Dir per i log di TensorBoard",
    )
    
    # --- DATI E FORMATI ---
    parser.add_argument(
        "--data_format",
        default="npy", 
        type=str,
        choices=["fits", "txt", "npy"],
        help="Formato del datacube di input"
    )
    parser.add_argument(
        "--object_id", # Modificato: da subject_id a object_id
        default="patch_000000",
        type=str,
        help="ID dell'oggetto celeste (es. nome galassia o ID catalogo)"
    )
    
    parser.add_argument(
    "--image_size", 
        type=int, 
        nargs='+', 
        default=[160, 224, 160], 
        help="Risoluzione target (D, H, W)"
    )

    parser.add_argument(
    "--z_channels",
        type=int,
        default=3,
        help="Canali del latente"
    )
    # --- PARAMETRI DI TRAINING GENERICI ---
    parser.add_argument(
        "--learning_rate",
        dest="learning_rate",
        default=1e-4, # Spesso astro richiede LR più bassi
        type=float,
    )
    parser.add_argument(
        "--batch_size",
        default=1, # I datacube 3D occupano molta VRAM
        type=int,
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
    )
    
    # --- OTTIMIZZAZIONE LATENTE (INVERSION) ---
    parser.add_argument(
        "--experiment_name",
        default="astro_restoration",
        type=str,
    )
    parser.add_argument(
        "--update_latent_variables",
        action="store_true",
        help="Se attivo, ottimizza il vettore latente z"
    )
    parser.add_argument(
        "--update_conditioning",
        action="store_true",
        help="Se attivo, ottimizza le condizioni fisiche (es. redshift)"
    )
    
    # --- CONDIZIONAMENTO ASTROFISICO ---
    
    parser.add_argument(
        "--update_frequency", 
        action="store_true",
        help="Ottimizza la stima della frequenza (f)"
    )
    
    parser.add_argument(
        "--update_flux_norm", 
        action="store_true",
        help="Ottimizza la normalizzazione del flusso"
    )
  

    # --- PERCEPTUAL LOSS & GEOMETRIA ---
    parser.add_argument(
        "--lambda_perc",
        default=0.01,
        type=float,
    )
    parser.add_argument(
        "--slicing_dim", # Modificato: da perc_dim (axial/coronal)
        default="spatial",
        type=str,
        # spatial = piano RA-DEC (somma sui canali o slice singola)
        # spectral = piano Posizione-Velocità (slice lungo RA o DEC)
        choices=["spatial", "spectral_ra", "spectral_dec"], 
        help="Direzione lungo cui calcolare la loss percettiva"
    )

    # --- DEGRADAZIONE / CORRUZIONE (Telescope effects) ---
    parser.add_argument(
        "--corruption",
        default="downsample",
        type=str,
        # beam_smear = convoluzione con PSF (Point Spread Function)
        # noise = rumore termico del ricevitore
        choices=["downsample", "mask", "beam_smear", "noise", "None"],
    )
    parser.add_argument(
        "--mask_id",
        default="survey_edge", # es. bordi del rilevatore
        type=str,
    )
    parser.add_argument(
        "--downsample_factor",
        default=4,
        type=int,
        choices=[2, 4, 8, 16],
        help="Fattore di riduzione risoluzione (binning spettrale o spaziale)"
    )
    
    # --- DDIM SAMPLING ---
    parser.add_argument(
        "--ddim_num_timesteps",
        default=10, # 250 è spesso eccessivo per test rapidi
        type=int,
    )
    parser.add_argument(
        "--ddim_eta",
        default=0.0, # 0.0 = Deterministico, 1.0 = DDPM standard
        type=float,
    )
    
    # --- LOSS EXTRA ---
    parser.add_argument(
        "--downsampling_loss",
        action="store_true",
        help="Forza la coerenza tra output HR e input LR"
    )
    parser.add_argument(
        "--mean_latent_vector",
        action="store_true",
        help="Inizia l'inversione dal vettore medio del dataset"
    )
    
    # --- ALTRO ---
    parser.add_argument(
        "--start_steps",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--num_steps",
        default=50,
        type=int,
    )
    parser.add_argument(
        "--prior_every",
        default=20,
        type=int,
    )
    parser.add_argument(
        "--n_samples",
        default=1, # Solitamente 1 datacube alla volta per limiti di memoria
        type=int,
    )

    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        type=str,
    )

    parser.add_argument(
        "--path_to_ddpm_checkpoint",    
        default="C:\\Modelli 3D\\InverseSR - Astro\\data\\trained_models_astro\\ddpm\\data\\model.pth",
        type=str,
        help="Path al checkpoint della DDPM pre-allenata"
    )
