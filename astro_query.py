import os
import requests
from requests.auth import HTTPBasicAuth

# 1. Configurazione Parametri e Credenziali
DOI = "10.25919/kqrt-pv24"
SAVE_DIR = "/leonardo_scratch/large/userexternal/gvitanza/Eridanus"
USERNAME = "giorgia.vitanza@inaf.it"
PASSWORD = "Romitici21!"

os.makedirs(SAVE_DIR, exist_ok=True)

# 2. API Endpoint per le collezioni CASDA
# Interroghiamo l'archivio CASDA per la lista dei file associati al DOI
api_url = f"https://casda.csiro.au/casda_data_access/data/collections/{DOI}/files"

print(f"Interrogazione CASDA API per il DOI: {DOI}...")
response = requests.get(api_url, auth=HTTPBasicAuth(USERNAME, PASSWORD))

if response.status_code != 200:
    # Se l'API restituisce errore, proviamo la rotta alternativa di ricerca TAP via HTTP POST
    print(f"⚠️ Chiamata API fallita ({response.status_code}). Tentativo tramite TAP POST diretto...")
    tap_url = "https://casda.csiro.au/casda_vo/tap/sync"
    query = f"SELECT access_url, filename FROM casda.observation_file WHERE collection_doi = '{DOI}'"
    
    response = requests.post(
        "https://casda.csiro.au/casda_vo/tap/sync",
        data={'REQUEST': 'doQuery', 'LANG': 'ADQL', 'FORMAT': 'csv', 'QUERY': query},
        auth=HTTPBasicAuth(USERNAME, PASSWORD)
    )

# 3. Parsing e Download dei File
if response.status_code == 200:
    lines = response.text.strip().split('\n')
    print(f"Trovati dati da scaricare. Inizio processo...")

    # Gestione sia per risposta JSON che CSV
    for line in lines[1:]: # Salta l'header
        if not line:
            continue
        parts = line.split(',')
        if len(parts) >= 1:
            file_url = parts[0].strip('"')
            filename = parts[1].strip('"') if len(parts) > 1 else file_url.split('/')[-1]
            out_path = os.path.join(SAVE_DIR, filename)

            print(f"\nScarico: {filename}")
            with requests.get(file_url, auth=HTTPBasicAuth(USERNAME, PASSWORD), stream=True) as r:
                r.raise_for_status()
                with open(out_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024): # Chunk da 1MB
                        f.write(chunk)
            print(f"✅ Completato: {filename}")
else:
    print(f"❌ Impossibile accedere all'archivio. Codice di stato: {response.status_code}")
    print(f"Risposta: {response.text[:200]}")