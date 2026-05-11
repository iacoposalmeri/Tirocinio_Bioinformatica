from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from utils_crc import *
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import silhouette_score, ConfusionMatrixDisplay, RocCurveDisplay, matthews_corrcoef
import pandas as pd

# aggiungere il parametro <<condizione_negativa = 'healthy'>> per il confronto fra pazienti malati di CRC e persone completamente sane
x, y_binary, metadati_finali_no_desease, metadati_esclusi = caricamento_pulizia_dati("Metadati_CRC_Dataset.csv", "Abbondanze_CRC_Dataset.csv")


X_train, X_test, y_train, y_test = train_test_split(x, y_binary, test_size=0.2, random_state=42, stratify=y_binary)

# CALCOLO PREVALENZA E TRASFORMAZIONE CLR

batteri_da_tenere = maschera_prevalenza(X_train, y_train, 0.05)

print(f"Prima del filtraggio al {0.05*100:.0f}%: {X_train.shape}")
X_train_filtrato_elb = filtraggio(X_train, batteri_da_tenere)
X_test_filtrato_elb = filtraggio(X_test, batteri_da_tenere)
print(f"Dopo il filtraggio al {0.05*100:.0f}%: {X_train_filtrato_elb.shape}")
X_train_clr_elb = trasformazione_clr(X_train_filtrato_elb)
X_test_clr_elb = trasformazione_clr(X_test_filtrato_elb)


X_train_scaled_elb, X_test_scaled_elb = standard_scaler(X_train_clr_elb, X_test_clr_elb)
 

pca_elb = PCA(n_components=50,random_state=42)

X_train_pca_elb = pca_elb.fit_transform(X_train_scaled_elb,y_train)
X_test_pca_elb = pca_elb.transform(X_test_scaled_elb)


# PCA E RICERCA DEL GOMITO

fig_elb, ax_elb = plt.subplots(figsize=(18,10))

variance_elb = pca_elb.explained_variance_ratio_ * 100
cumulative_variance_elb = np.cumsum(variance_elb)

ax_elb.bar(range(1,len(variance_elb)+1), variance_elb, alpha=0.6, align='center', label='Varianza Singola', color='#3498db')
ax_elb.step(range(1,len(cumulative_variance_elb)+1),cumulative_variance_elb, where='mid', label='Varianza Cumulata', color='#e74c3c', marker='o', markersize=4)

ax_elb.set_ylabel('Varianza Spiegata (%)', fontsize=12)
ax_elb.set_xlabel('Indice della Componente Principale (PC)', fontsize=12)
ax_elb.set_title('Scree Plot: Ricerca del Gomito (Elbow Method)', fontsize=14, fontweight='bold')
ax_elb.set_xticks(np.arange(0, 51, step=5)) 
ax_elb.legend(loc='center right')
ax_elb.grid(True, linestyle='--', alpha=0.5)

fig_elb.savefig("Explained_Variance_PCA_Aitchison.png", dpi=300, bbox_inches='tight')

score_elb = silhouette_score(X_train_pca_elb[:, :10], y_train)
print(f"Il silhouette score dell'Atichison PCA e' {score_elb}")

fig_s, ax_s = plt.subplots(figsize = (18,10))

sns.scatterplot(
    x = X_train_pca_elb[:,1],
    y = X_train_pca_elb[:,2],
    hue = y_train.map({0:'Control', 1 : 'CRC'}),
    palette = {'Control' : "#F46E27", 'CRC' : "#1C5CDC"},
    ax = ax_s,
    alpha = 0.7,
    s = 50
)

var_pc1 = pca_elb.explained_variance_ratio_[0] * 100
var_pc2 = pca_elb.explained_variance_ratio_[1] * 100

ax_s.set_title(f"Cutoff {0.05*100:.0f}%", fontweight='bold', fontsize=12)
ax_s.set_xlabel(f"PC1 ({var_pc1:.1f}%)")
ax_s.set_ylabel(f"PC2 ({var_pc2:.1f}%)")

fig_s.savefig("Grid_PCA_Aitchison.png", dpi=300, bbox_inches='tight')

loadings_elb = pca_elb.components_[0]

df_loadings = pd.DataFrame({
    'Batterio' : X_train_clr_elb.columns,
    'Peso_pc1' : loadings_elb
})

df_loadings['Forza_Assoluta']=df_loadings['Peso_pc1'].abs()

top_15_bacteria = df_loadings.sort_values(by = 'Forza_Assoluta', ascending = False).head(15)


fig_s, ax_s = plt.subplots(figsize=(18,10))
sns.barplot(data = top_15_bacteria, x = 'Forza_Assoluta', y = 'Batterio', hue='Batterio', legend=False, ax = ax_s, palette = 'viridis')
ax_s.set_ylabel("Specie Batterica", fontsize=12)

fig_s.tight_layout()
fig_s.savefig("Loadings_PC1_Aitchison.png", dpi=300, bbox_inches='tight')

##CONFUSION MATRIX
n_comp_cm=15
X_train_cm = X_train_pca_elb[:, :n_comp_cm]
X_test_cm = X_test_pca_elb[:, :n_comp_cm]

RF_cm = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
RF_cm.fit(X_train_cm,y_train)

y_pred_cm = RF_cm.predict(X_test_cm)

fig_m, ax_m = plt.subplots(figsize=(8,6))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred_cm,
    display_labels=["Control", "CRC"],
    cmap='Blues',
    ax=ax_m,
    colorbar=False
)

ax_m.set_title("Matrice di Confusione: Random Forest (Aitchison 5%)", fontsize=14, fontweight='bold')

fig_m.savefig("Confusion_Matrix_Vincitore.png", dpi=300, bbox_inches='tight')

#ROC
fig_r, ax_r  = plt.subplots(figsize = (18,10))

RocCurveDisplay.from_estimator(
    RF_cm,
    X_test_cm,
    y_test,
    ax=ax_r,
    name='Random Forest (Aitchison 5%)',
    curve_kwargs={'color': 'darkorange'}
)

ax_r.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

ax_r.set_title("Curva ROC: Random Forest (Aitchison 5%)", fontsize=14, fontweight='bold')
ax_r.grid(True, linestyle='--', alpha=0.5)

fig_r.savefig("ROC_Curve_Vincitore.png", dpi=300, bbox_inches='tight')


#MCC
print(f"Il Coefficiente di Correlazione di Matthews (MCC) e': {matthews_corrcoef(y_test,y_pred_cm):.6f}")

