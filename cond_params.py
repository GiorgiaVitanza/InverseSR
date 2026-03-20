import numpy as np

catalogue_path = "data/inputs/sky_dev_truthcat_v2.txt"

# Carica i dati (salta le righe di intestazione se presenti con skiprows=1)
data = np.loadtxt(catalogue_path, skiprows=1)
col_names = np.loadtxt(catalogue_path, max_rows=1, dtype=str)  # Leggi la prima riga per i nomi delle colonne

# Ora puoi accedere alle colonne (usando l'indice della colonna)
# data[:, 4] prende tutte le righe della quinta colonna
cols_to_check = [3, 4, 7, 8]

for i, j in zip(cols_to_check, col_names[cols_to_check]):
    col = data[:, i]
    print(f"Colonna {j} -> Max: {col.max()}, Min: {col.min()}, Mean: {col.mean()}, Std: {col.std()}")