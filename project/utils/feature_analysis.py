import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# 1. Caricamento dei dati
data_path = "/leonardo_scratch/large/userexternal/gvitanza/InverseSR/data/inputs/128x128x128_stride128/train/train_catalog.csv"

# Carichiamo l'intero dataset che hai fornito
df = pd.read_csv(data_path, sep=',')

# Rimuoviamo l'ID che è solo un indice e non una feature fisica
df_features = df.drop(columns=["patch_id", "source_id"])

print("--- Anteprima del Dataset ---")
print(df_features.head())
print("\n--- Statistiche Descrittive ---")
print(df_features.describe())

# ---------------------------------------------------------
# METODO 1: Matrice di Correlazione
# ---------------------------------------------------------
plt.figure(figsize=(10, 8))
correlation_matrix = df_features.corr(method="spearman")  # Spearman rileva anche relazioni non lineari monotone
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
plt.title("Matrice di Correlazione di Spearman")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# METODO 2: Feature Importance con Random Forest
# Poniamo come esempio di voler predire 'w20' (larghezza del profilo a 20%),
# che spesso è la feature di condizionamento/output cruciale.
# Cambia il 'target' se hai un'altra variabile in mente.
# ---------------------------------------------------------
target_variable = "w20"

if target_variable in df_features.columns:
    X = df_features.drop(columns=[target_variable])
    y = df_features[target_variable]

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title(f"Feature Importance per la predizione di [{target_variable}] (Random Forest)")
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), [X.columns[i] for i in indices], rotation=45)
    plt.tight_layout()
    plt.show()

    print(f"\n--- Classifica Feature Importance per {target_variable} ---")
    for start, idx in enumerate(indices):
        print(f"{start + 1}. {X.columns[idx]}: {importances[idx]:.4f}")

# ---------------------------------------------------------
# METODO 3: Analisi delle Componenti Principali (PCA)
# Capire quali variabili pesano di più sulla varianza totale del sistema
# ---------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_features)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Carico (loadings) delle feature sulle prime due componenti fisiche
loadings = pd.DataFrame(
    pca.components_.T, columns=["PC1", "PC2"], index=df_features.columns
)

print("\n--- Carico delle Feature sulle Componenti Principali (PCA) ---")
print(loadings)

# Grafico della varianza spiegata
plt.figure(figsize=(8, 5))
plt.bar(["PC1", "PC2"], pca.explained_variance_ratio_, color="darkcyan")
plt.ylabel("Rapporto di Varianza Spiegata")
plt.title("Varianza Spiegata dalle prime due Componenti Principali")
plt.show()