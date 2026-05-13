from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from utils_crc import * 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from skbio.stats.ordination import pcoa, pcoa_biplot
from skbio.diversity import beta_diversity

# aggiungere il parametro <<condizione_negativa = 'healthy'>> per il confronto fra pazienti malati di CRC e persone completamente sane
x, y_binary, metadati_finali_no_desease, metadati_esclusi = caricamento_pulizia_dati("Metadati_CRC_Dataset.csv", "Abbondanze_CRC_Dataset.csv")



X_train, X_test, y_train, y_test = train_test_split(x, y_binary, test_size=0.2, random_state=42, stratify=y_binary)


cutoff = 0.1


batteri_da_tenere_ex = maschera_prevalenza(X_train, y_train, cutoff)

print(f"Prima del filtraggio al {cutoff*100:.0f}%: {X_train.shape}")
X_train_filtrato_ex = filtraggio(X_train, batteri_da_tenere_ex)
#X_test_filtrato_ex = filtraggio(X_test, batteri_da_tenere_ex)

print(f"Dopo il filtraggio al {cutoff*100:.0f}%: {X_train_filtrato_ex.shape}")


#BLOCCO PCoA

dm = beta_diversity('braycurtis', X_train_filtrato_ex.values, X_train_filtrato_ex.index.tolist())

pcoa_res = pcoa(dm)

biplot_res = pcoa_biplot(pcoa_res,X_train_filtrato_ex)

loadings = biplot_res.features.copy()

pc1 = loadings.columns[0]

top_bacteria_inorder = loadings.sort_values(by=pc1, ascending=False)

print("\n--- I 10 BATTERI CHE SPINGONO VERSO DESTRA (PC1 Positiva) ---")
print(top_bacteria_inorder.head(10)[pc1])

print("\n--- I 10 BATTERI CHE SPINGONO VERSO SINISTRA (PC1 Negativa) ---")
print(top_bacteria_inorder.tail(10)[pc1])

# GENERAZIONE GRAFICO

df_samples = biplot_res.samples[['PC1','PC2']].copy()

df_samples['Condition'] = y_train.map({0: 'Control', 1: 'CRC'})

fig, ax = plt.subplots(1,2,figsize=(20, 10))


sns.scatterplot(
    x='PC1', y='PC2',
    hue='Condition',
    palette={'Control': "#EE752A", 'CRC': "#02CAF1"},
    data=df_samples,
    s=70, alpha=0.7, edgecolor='k', ax=ax[0]
)

var_pc1 = biplot_res.proportion_explained['PC1'] * 100
var_pc2 = biplot_res.proportion_explained['PC2'] * 100

ax[0].set_title(f"PCoA Biplot (Bray-Curtis) - Cutoff {cutoff*100:.0f}% - Patients", fontweight='bold', fontsize=16)
ax[0].set_xlabel(f'Varianza Spiegata PCoA 1: {var_pc1:.2f}%', fontsize=12)
ax[0].set_ylabel(f'Varianza Spiegata PCoA 2: {var_pc2:.2f}%', fontsize=12)

x_min, x_max = ax[0].get_xlim()
y_min, y_max = ax[0].get_ylim()
ax[0].set_xlim(x_min * 1.2, x_max * 1.2)
ax[0].set_ylim(y_min * 1.2, y_max * 1.2)

ax[0].axhline(0, color='grey', linestyle='--', alpha=0.5, zorder=1)
ax[0].axvline(0, color='grey', linestyle='--', alpha=0.5, zorder=1)
ax[0].grid(True, linestyle=':', alpha=0.4)

top_pos = top_bacteria_inorder[[pc1]].head(10).copy()
top_neg = top_bacteria_inorder[[pc1]].tail(10).copy()
top_20 = pd.concat([top_pos, top_neg])

df_top20 = top_20.reset_index()
df_top20.columns = ['Bacteria', 'PC1_Weight']

df_top20['Bacteria'] = df_top20['Bacteria'].str.replace('species:', '', regex=False)

color_bars = ['#D62728' if peso > 0 else '#1F77B4' for peso in df_top20['PC1_Weight']]

sns.barplot(
    x='PC1_Weight', 
    y='Bacteria',
    data = df_top20,
    hue='Bacteria',
    palette=color_bars,
    legend=False,
    edgecolor='black',
    ax=ax[1]
)

ax[1].set_title("Top 20 Biomarkers (Impact on PC1)", fontweight='bold', fontsize=16)
ax[1].set_xlabel("Loading PC1", fontsize=12)
ax[1].set_ylabel("") 


ax[1].axvline(0, color='black', linewidth=1.5, zorder=3)
ax[1].grid(True, axis='x', linestyle=':', alpha=0.5)

fig.tight_layout()
fig.savefig("PCoA_Biplot_Cutoff_10.png", dpi=300, bbox_inches='tight')