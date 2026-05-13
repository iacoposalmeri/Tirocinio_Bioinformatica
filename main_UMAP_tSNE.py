import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import umap
from sklearn.manifold import TSNE
from sklearn.cluster import HDBSCAN
from sklearn.model_selection import train_test_split
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from skbio.diversity import beta_diversity
from utils_crc import * 

print("Caricamento e pulizia dati in corso...")
x, y_binary, metadati_finali_no_desease, metadati_esclusi = caricamento_pulizia_dati("Metadati_CRC_Dataset.csv", "Abbondanze_CRC_Dataset.csv")

SEED = 42
X_train, X_test, y_train, y_test = train_test_split(x, y_binary, test_size=0.2, random_state=SEED, stratify=y_binary)


cutoff = 0.05
print(f"Dimensione originale train: {X_train.shape}")

batteri_da_tenere = maschera_prevalenza(X_train, y_train, cutoff)
X_train_filtrato = filtraggio(X_train, batteri_da_tenere)
X_test_filtrato = filtraggio(X_test, batteri_da_tenere)
print(f"Dimensione dopo filtraggio al {cutoff*100:.0f}%: {X_train_filtrato.shape}")


print("\n--- CLR ---")
X_train_clr = trasformazione_clr(X_train_filtrato)
X_test_clr = trasformazione_clr(X_test_filtrato)

X_train_scaled, X_test_scaled = standard_scaler(X_train_clr, X_test_clr)

print("Calcolo t-SNE (CLR)...")
tsne_clr = TSNE(n_components=2, metric='euclidean', perplexity=30, random_state=SEED)
tsne_clr_res = tsne_clr.fit_transform(X_train_scaled)

print("Calcolo UMAP (CLR)...")
umap_clr = umap.UMAP(n_components=2, metric='euclidean', n_neighbors=15, min_dist=0.1, random_state=SEED)
umap_clr_res = umap_clr.fit_transform(X_train_scaled)


print("\n--- Bray-Curtis ---")
print("Calcolo matrice delle distanze Bray-Curtis...")
dm = beta_diversity('braycurtis', X_train_filtrato.values, X_train_filtrato.index.tolist())

print("Calcolo t-SNE (Bray-Curtis)...")
tsne_bc = TSNE(n_components=2, metric='precomputed', init='random', perplexity=30, random_state=SEED)
tsne_bc_res = tsne_bc.fit_transform(dm.data)

print("Calcolo UMAP (Bray-Curtis)...")
umap_bc = umap.UMAP(n_components=2, metric='precomputed', n_neighbors=15, min_dist=0.1, random_state=SEED)
umap_bc_res = umap_bc.fit_transform(dm.data)

print("\n--- Ricerca Cluster con HDBSCAN (Alta Dimensionalità) ---")


print("Compressione a 15 dimensioni per HDBSCAN...")
umap_ml = umap.UMAP(n_components=15, metric='precomputed', n_neighbors=15, min_dist=0.0, random_state=SEED)
umap_15d_res = umap_ml.fit_transform(dm.data)

hdb = HDBSCAN(min_cluster_size=20, min_samples=5)
cluster_labels = hdb.fit_predict(umap_15d_res)
