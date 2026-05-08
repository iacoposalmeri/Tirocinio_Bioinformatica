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


# aggiungere il parametro <<condizione_negativa = 'healthy'>> per il confronto fra pazienti malati di CRC e persone completamente sane
x, y_binary, metadati_finali_no_desease, metadati_esclusi = caricamento_pulizia_dati("Metadati_CRC_Dataset.csv", "Abbondanze_CRC_Dataset.csv")



X_train, X_test, y_train, y_test = train_test_split(x, y_binary, test_size=0.2, random_state=42, stratify=y_binary)


# RICERCA DEL GOMITO

batteri_da_tenere_ex = maschera_prevalenza(X_train, y_train, 0.1)

print(f"Prima del filtraggio al {0.1*100:.0f}%: {X_train.shape}")
X_train_filtrato_ex = filtraggio(X_train, batteri_da_tenere_ex)
X_test_filtrato_ex = filtraggio(X_test, batteri_da_tenere_ex)
X_train_scaled_ex, X_test_scaled_ex = standard_scaler(X_train_filtrato_ex, X_test_filtrato_ex)
print(f"Dopo il filtraggio al {0.1*100:.0f}%: {X_train_filtrato_ex.shape}")


#BLOCCO PCA
pca_ex = PCA(n_components=50,random_state=42)

X_train_pca_ex = pca_ex.fit_transform(X_train_scaled_ex) 
X_test_pca_ex = pca_ex.transform(X_test_scaled_ex)

variance = pca_ex.explained_variance_ratio_ * 100
cumulativa_var = np.cumsum(variance)

fig1, ax = plt.subplots(figsize=(18, 10))

ax.bar(range(1,len(variance) +1), variance, alpha=0.6, align='center', label='Varianza Singola', color='#3498db')
ax.step(range(1, len(cumulativa_var) + 1), cumulativa_var, 
        where='mid', label='Varianza Cumulata', color='#e74c3c', marker='o', markersize=4)

ax.set_ylabel('Varianza Spiegata (%)', fontsize=12)
ax.set_xlabel('Indice della Componente Principale (PC)', fontsize=12)
ax.set_title('Scree Plot: Ricerca del Gomito (Elbow Method)', fontsize=14, fontweight='bold')
ax.set_xticks(np.arange(0, 51, step=5)) 
ax.legend(loc='center right')
ax.grid(True, linestyle='--', alpha=0.5)

fig1.tight_layout()

fig1.savefig("Explained_Variance_PCA.png", dpi=300, bbox_inches='tight')


components = pca_ex.components_[0]

df_loadings = pd.DataFrame({
    'Batterio' : X_train_filtrato_ex.columns,
    'Peso_pc1' : components
})

df_loadings['Forza_Assoluta']=df_loadings['Peso_pc1'].abs()

top_15_bacteria = df_loadings.sort_values(by = 'Forza_Assoluta', ascending = False).head(15)


fig2, ax2 = plt.subplots(figsize=(18,10))
sns.barplot(data = top_15_bacteria, x = 'Forza_Assoluta', y = 'Batterio', hue='Batterio', legend=False, ax = ax2, palette = 'viridis')
ax2.set_ylabel("Specie Batterica", fontsize=12)

fig2.tight_layout()
fig2.savefig("Loadings_PC1_Tradizionale.png", dpi=300, bbox_inches='tight')

score = silhouette_score(X_train_pca_ex, y_train)
print(f"Il Silhouette Score della PCA Tradizionale è: {score:.4f}")