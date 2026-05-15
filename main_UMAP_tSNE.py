import warnings
# Ignora gli avvisi di UMAP (n_jobs e precomputed)
warnings.filterwarnings('ignore', category=UserWarning)
# Ignora gli avvisi di scikit-learn sui futuri aggiornamenti (copy)
warnings.filterwarnings('ignore', category=FutureWarning)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import umap
from sklearn.manifold import TSNE
from sklearn.cluster import HDBSCAN
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, matthews_corrcoef
from skbio.diversity import beta_diversity
from utils_crc import * 
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from xgboost import XGBClassifier

print("Caricamento e pulizia dei dati")
x, y_binary, metadati_finali_no_desease, metadati_esclusi = caricamento_pulizia_dati("Metadati_CRC_Dataset.csv", "Abbondanze_CRC_Dataset.csv",condizione_negativa='control')

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
umap_clr = umap.UMAP(n_components=2, metric='euclidean', n_neighbors=5, min_dist=0.0, random_state=SEED)
umap_clr_res = umap_clr.fit_transform(X_train_scaled)


print("\n--- Bray-Curtis ---")
print("Calcolo matrice delle distanze Bray-Curtis...")
dm = beta_diversity('braycurtis', X_train_filtrato.values, X_train_filtrato.index.tolist())

print("Calcolo t-SNE (Bray-Curtis)...")
tsne_bc = TSNE(n_components=2, metric='precomputed', init='random', perplexity=30, random_state=SEED)
tsne_bc_res = tsne_bc.fit_transform(dm.data)

print("Calcolo UMAP (Bray-Curtis)...")
umap_bc = umap.UMAP(n_components=2, metric='precomputed', n_neighbors=15, min_dist=0.0, random_state=SEED)
umap_bc_res = umap_bc.fit_transform(dm.data)

print("\n--- Ricerca Cluster con HDBSCAN (Alta Dimensionalità) ---")


print("Compressione a 15 dimensioni per HDBSCAN...")
umap_15d_clr = umap.UMAP(n_components=15, metric='euclidean', n_neighbors=5, min_dist=0.0, random_state=SEED)
umap_clr_15d_res = umap_15d_clr.fit_transform(X_train_clr)

umap_15d_bc = umap.UMAP(n_components=15, metric='precomputed', n_neighbors=15, min_dist=0.0, random_state=SEED)
umap_bc_15d_res = umap_15d_bc.fit_transform(dm.data)

hdb_clr = HDBSCAN(min_cluster_size=10, min_samples=5)
cluster_labels_clr = hdb_clr.fit_predict(umap_clr_15d_res)

hdb_bc = HDBSCAN(min_cluster_size=20, min_samples=5)
cluster_labels_bc = hdb_bc.fit_predict(umap_bc_15d_res)

y_train_aligned = y_train.reset_index(drop=True)

n_clusters_clr = len(set(cluster_labels_clr) - {-1})
n_clusters_bc = len(set(cluster_labels_bc) - {-1})

n_noise_clr = list(cluster_labels_clr).count(-1)
n_noise_bc = list(cluster_labels_bc).count(-1)

print(f"HDBSCAN su CLR: Trovati {n_clusters_clr} cluster. Pazienti rumore: {n_noise_clr}")
print(f"HDBSCAN su BC : Trovati {n_clusters_bc} cluster. Pazienti rumore: {n_noise_bc}")

print("Tabella di Contingenza CLR:")
tabella_clr = pd.crosstab(cluster_labels_clr, y_train_aligned)
tabella_clr.columns = ['Control','CRC']
tabella_clr.index.name = 'Cluster ID'
print(tabella_clr)

print("Tabella di Contingenza Bray-Curtis:")
tabella_bc = pd.crosstab(cluster_labels_bc, y_train_aligned)
tabella_bc.columns = ['Control','CRC']
tabella_bc.index.name = 'Cluster ID'
print(tabella_bc)

# QUAD-PLOT UMAP E t-SNE

fig1, ax1 = plt.subplots(2, 2, figsize=(18, 18))

labels_cond = y_train_aligned.map({0: 'Control', 1: 'CRC'})
palette_cond = {'Control' : "#FF9500", 'CRC' : "#00F7FF"}

sns.scatterplot(
    x = tsne_clr_res[:,0],
    y = tsne_clr_res[:,1],
    hue = labels_cond,
    palette = palette_cond,
    ax = ax1[0,0],
    alpha = 0.7,
    s = 50
)

sns.scatterplot(
    x = umap_clr_res[:,0],
    y = umap_clr_res[:,1],
    hue = labels_cond,
    palette = palette_cond,
    ax = ax1[0,1],
    alpha = 0.7,
    s = 50
)

sns.scatterplot(
    x = tsne_bc_res[:,0],
    y = tsne_bc_res[:,1],
    hue = labels_cond,
    palette = palette_cond,
    ax = ax1[1,0],
    alpha = 0.7,
    s = 50
)

sns.scatterplot(
    x = umap_bc_res[:,0],
    y = umap_bc_res[:,1],
    hue = labels_cond,
    palette = palette_cond,
    ax = ax1[1,1],
    alpha = 0.7,
    s = 50
)

ax1[0, 0].set_title("t-SNE (CLR)", fontsize=14, fontweight='bold')
ax1[0, 0].grid(True, linestyle='--', alpha=0.5)

ax1[0, 1].set_title("UMAP (CLR)", fontsize=14, fontweight='bold')
ax1[0, 1].grid(True, linestyle='--', alpha=0.5)

ax1[1, 0].set_title("t-SNE (Bray-Curtis)", fontsize=14, fontweight='bold')
ax1[1, 0].grid(True, linestyle='--', alpha=0.5)

ax1[1, 1].set_title("UMAP (Bray-Curtis)", fontsize=14, fontweight='bold')
ax1[1, 1].grid(True, linestyle='--', alpha=0.5)

fig1.suptitle("Analisi Esplorativa: Sani vs CRC", fontsize=18, fontweight='bold', y=0.95)

fig1.tight_layout()
fig1.savefig("QuadPlot_Clinico.png", dpi=300, bbox_inches='tight')

# BIPLOT HDBSCAN

fig2, ax2 = plt.subplots(1, 2, figsize=(20, 9))

def disegna_cluster(dati_2d, etichette_cluster, titolo, asse):
    # Trasformiamo le etichette in stringhe pulite
    cluster_nomi = [f"Cluster {c}" if c != -1 else "Rumore (-1)" for c in etichette_cluster]
    df_temp = pd.DataFrame({'Dim1': dati_2d[:, 0], 'Dim2': dati_2d[:, 1], 'ID': cluster_nomi})

    sns.scatterplot(data=df_temp[df_temp['ID'] == 'Rumore (-1)'], 
                    x='Dim1', y='Dim2', color='lightgrey', s=30, alpha=0.5, ax=asse, label='Rumore (-1)')
    
    sns.scatterplot(data=df_temp[df_temp['ID'] != 'Rumore (-1)'], 
                    x='Dim1', y='Dim2', hue='ID', palette='tab10', s=80, alpha=0.9, edgecolor='black', ax=asse)
    
    asse.set_title(titolo, fontweight='bold', fontsize=16)
    asse.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    asse.grid(True, linestyle='--', alpha=0.5)


disegna_cluster(umap_clr_res, cluster_labels_clr, f"Cluster CLR (Trovati: {n_clusters_clr})", ax2[0])
disegna_cluster(umap_bc_res, cluster_labels_bc, f"Cluster Bray-Curtis (Trovati: {n_clusters_bc})", ax2[1])

fig2.tight_layout()
fig2.savefig("BiPlot_Clusters.png", dpi=300, bbox_inches='tight')


RF = RandomForestClassifier(random_state=SEED, class_weight='balanced')
XGB = XGBClassifier(n_jobs=-1)
SVM = svm.SVC(kernel="rbf", class_weight = 'balanced', random_state=SEED)

cv_strategy = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)

measures = {
    'accuracy': 'accuracy',
    'f1': 'f1',
    'recall': 'recall',
    'precision': 'precision',
    'mcc': make_scorer(matthews_corrcoef)
}

rf_res = cross_validate(RF, umap_clr_15d_res, y_train_aligned, cv=cv_strategy, scoring=measures)
xgb_res = cross_validate(XGB, umap_clr_15d_res, y_train_aligned, cv=cv_strategy, scoring=measures)
svm_res = cross_validate(SVM, umap_clr_15d_res, y_train_aligned, cv=cv_strategy, scoring=measures)

print("\n" + "="*55)
print("   RISULTATI PREDITTIVI: CROSS-VALIDATION (10-Fold)")
print("           (Addestramento su UMAP CLR 15D)")
print("="*55)

# --- Risultati Random Forest ---
print("\n--- Modello 1: RANDOM FOREST ---")
print(f"Accuratezza Media   : {rf_res['test_accuracy'].mean() * 100:.2f}%")
print(f"Sensibilità (Recall): {rf_res['test_recall'].mean() * 100:.2f}%  <-- Capacità di trovare i veri Malati")
print(f"Precisione Media    : {rf_res['test_precision'].mean() * 100:.2f}%")
print(f"F1-Score Medio      : {rf_res['test_f1'].mean() * 100:.2f}%  <-- (Metrica Globale)")
print(f"MCC Medio           : {rf_res['test_mcc'].mean():.3f}  <-- (0 = Casuale, 1 = Perfetto)")

# --- Risultati XGB ---
print("\n--- Modello 2: XGB ---")
print(f"Accuratezza Media   : {xgb_res['test_accuracy'].mean() * 100:.2f}%")
print(f"Sensibilità (Recall): {xgb_res['test_recall'].mean() * 100:.2f}%  <-- Capacità di trovare i veri Malati")
print(f"Precisione Media    : {xgb_res['test_precision'].mean() * 100:.2f}%")
print(f"F1-Score Medio      : {xgb_res['test_f1'].mean() * 100:.2f}%  <-- (Metrica Globale)")
print(f"MCC Medio           : {xgb_res['test_mcc'].mean():.3f}  <-- (0 = Casuale, 1 = Perfetto)")

# --- Risultati SVM ---
print("\n--- Modello 3: SUPPORT VECTOR MACHINE (SVM) ---")
print(f"Accuratezza Media   : {svm_res['test_accuracy'].mean() * 100:.2f}%")
print(f"Sensibilità (Recall): {svm_res['test_recall'].mean() * 100:.2f}%  <-- Capacità di trovare i veri Malati")
print(f"Precisione Media    : {svm_res['test_precision'].mean() * 100:.2f}%")
print(f"F1-Score Medio      : {svm_res['test_f1'].mean() * 100:.2f}%  <-- (Metrica Globale)")
print(f"MCC Medio           : {svm_res['test_mcc'].mean():.3f}  <-- (0 = Casuale, 1 = Perfetto)")

print("\n" + "="*55)