import os
from pathlib import Path

# --- 1. RILEVAMENTO AMBIENTE (Cluster vs Locale) ---
# Controlla se siamo su un cluster HPC (SLURM) o sul PC locale
# SLURM_TMPDIR è una variabile standard in molti cluster
IS_CLUSTER = False
TMP_DIR = os.environ.get("SLURM_TMPDIR")

if TMP_DIR:
    IS_CLUSTER = True

# --- 2. DEFINIZIONE PERCORSI (PATHS) ---
# Qui definiamo dove il codice deve cercare i file FITS e dove salvare i modelli
# ROOT_DIR calcola la cartella principale del progetto automaticamente
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

if IS_CLUSTER:
    # Percorsi veloci su disco temporaneo del cluster
    WORK_DIR = Path(str(TMP_DIR)).resolve() / "work"
    
    INPUT_FOLDER = WORK_DIR / "inputs"
    MASK_FOLDER = WORK_DIR / "inputs" / "masks"
    
    # Cartelle per i pesi dei modelli pre-addestrati
    PRETRAINED_MODEL_FOLDER = WORK_DIR / "trained_models_astro"
    PRETRAINED_MODEL_DDPM_PATH = PRETRAINED_MODEL_FOLDER / "trained_models_ddpm_100" / "crossattn"
    PRETRAINED_MODEL_VAE_PATH = PRETRAINED_MODEL_FOLDER / "trained_models_astro_400" / "vae_400"
    
    # Output
    OUTPUT_FOLDER = WORK_DIR / "outputs"

else:
    # Percorsi locali (sul tuo PC/Server Lab)
    DATA_ROOT = ROOT_DIR / "data"
    
    INPUT_FOLDER = DATA_ROOT / "inputs"
    INPUT_FOLDER_PATCHES = INPUT_FOLDER / "128x128x128_stride128" / "npy_patches" # Se usi patch pre-estratti
    INPUT_FOLDER_TEST = DATA_ROOT / "test_5d"
    MASK_FOLDER = DATA_ROOT / "masks"
    
    # Dataset specifici (Esempio: ALMA, LOFAR, Simulazioni TNG)
    SURVEY_DATA_FOLDER = DATA_ROOT / "survey_data" 
    
    PRETRAINED_MODEL_FOLDER = DATA_ROOT / "trained_models_astro"
    
    PRETRAINED_MODEL_DDPM_PATH = PRETRAINED_MODEL_FOLDER  / "trained_models_ddpm_100" / "crossattn"
    PRETRAINED_MODEL_DECODER_PATH = PRETRAINED_MODEL_FOLDER / "trained_models_astro_400" / "decoder_400"
    PRETRAINED_MODEL_VAE_PATH = PRETRAINED_MODEL_FOLDER / "trained_models_astro_400" / "vae_400"
    
    # Percorso per un eventuale modello di feature extraction (es. per loss percettiva)
    # Nota: VGG16 è per immagini 2D. Se usi cubi 3D, potresti non usarlo o usare una 3D-ResNet.
    PRETRAINED_MODEL_VGG_PATH = PRETRAINED_MODEL_FOLDER / "vgg" / "vgg16_slim_astro.pth"

    OUTPUT_FOLDER = DATA_ROOT / "outputs" 
    FIGURES_FOLDER = DATA_ROOT / "figures" # Ex thesis_imgs
    FINAL_RESULTS_FOLDER = OUTPUT_FOLDER / "final_results"

# --- 3. DIMENSIONI DATI (CRUCIALE) ---

# IMAGE_SHAPE: [Batch, Channels, Depth (Freq/Vel), Height (Dec), Width (RA)]
# Nota: I modelli di diffusione richiedono dimensioni fisse (spesso multipli di 32 o 64).

IMAGE_SHAPE = [1, 3, 128, 128, 128] 

# LATENT_SHAPE: La dimensione compressa nel "Latent Space" del VAE.
# Dipende dal fattore di downsampling del tuo modello (spesso f=4 o f=8).
# Esempio: Se IMAGE_SHAPE è 128^3 e il downsampling è 4 -> 128/4 = 32.
LATENT_SHAPE = [1, 3, int(128/4), int(128/4), int(128/4)] 


# --- 4. LISTA OGGETTI (TARGETS) ---
# Invece di una lista hardcoded di 100 pazienti, qui puoi mettere
# gli oggetti specifici su cui vuoi testare il modello.
# Oppure lasciare una lista vuota e caricarli dinamicamente nello script principale.

TEST_OBJECTS = [
    "patch_000007",
    "patch_000008",
    "patch_000009",
    # Aggiungi qui i tuoi nomi file (senza estensione .npy)
]

# Helper per caricare tutti i file se la lista sopra è vuota
def get_all_objects(folder_path):
    return [f.stem for f in Path(folder_path).glob("*.fits")]
