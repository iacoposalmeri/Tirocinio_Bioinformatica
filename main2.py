from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from funzioni_crc2 import * 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

cutoffs = [0.03, 0.05, 0.07, 0.1, 0.15, 0.2]

x, y_binary, metadati_finali_no_desease, metadati_esclusi = caricamento_pulizia_dati("Metadati_CRC_Dataset.csv", "Abbondanze_CRC_Dataset.csv")



X_train, X_test, y_train, y_test = train_test_split(x, y_binary, test_size=0.2, random_state=42, stratify=y_binary)


#Cerco il gomito

batteri_da_tenere_ex = maschera_prevalenza(X_train, y_train, 0.1)

'''
X_train_clr = trasformazione_clr(X_train)
X_test_clr = trasformazione_clr(X_test)
print(f"Prima del filtraggio al {cutoff*100:.0f}%: {X_train_clr.shape}")
X_train_filtrato = filtraggio(X_train_clr, batteri_da_tenere)
X_test_filtrato = filtraggio(X_test_clr, batteri_da_tenere)
X_train_scaled, X_test_scaled = standard_scaler(X_train_filtrato, X_test_filtrato)
print(f"Dopo il filtraggio al {cutoff*100:.0f}%: {X_train_filtrato.shape}")
'''

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


""" fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
axes = axes.flatten()

for i, cutoff in enumerate(cutoffs):

    batteri_da_tenere = maschera_prevalenza(X_train, y_train, cutoff)

    '''
    X_train_clr = trasformazione_clr(X_train)
    X_test_clr = trasformazione_clr(X_test)
    print(f"Prima del filtraggio al {cutoff*100:.0f}%: {X_train_clr.shape}")
    X_train_filtrato = filtraggio(X_train_clr, batteri_da_tenere)
    X_test_filtrato = filtraggio(X_test_clr, batteri_da_tenere)
    X_train_scaled, X_test_scaled = standard_scaler(X_train_filtrato, X_test_filtrato)
    print(f"Dopo il filtraggio al {cutoff*100:.0f}%: {X_train_filtrato.shape}")
    '''

    print(f"Prima del filtraggio al {cutoff*100:.0f}%: {X_train.shape}")
    X_train_filtrato = filtraggio(X_train, batteri_da_tenere)
    X_test_filtrato = filtraggio(X_test, batteri_da_tenere)
    X_train_scaled, X_test_scaled = standard_scaler(X_train_filtrato, X_test_filtrato)
    print(f"Dopo il filtraggio al {cutoff*100:.0f}%: {X_train_filtrato.shape}")


    #BLOCCO PCA
    pca = PCA(n_components=10,random_state=42)

    X_train_pca = pca.fit_transform(X_train_scaled,y_train)
    X_test_pca = pca.transform(X_test_scaled)

    varianza_totale = pca.explained_variance_ratio_.sum() * 100
    print(f"La PCA ha compresso i dati in {pca.n_components_} dimensioni salvando il {varianza_totale:.1f}% dell'informazione.")

    sns.scatterplot(
        x = X_train_pca[:, 0],
        y = X_train_pca[:, 1],
        hue = y_train.map({0: 'Control', 1: 'CRC'}),
        palette = {'Control' : "#EE752A", 'CRC' : "#02CAF1"},
        ax = axes[i],
        alpha = 0.7,
        s = 50
    )

    var_pc1 = pca.explained_variance_ratio_[0] * 100
    var_pc2 = pca.explained_variance_ratio_[1] * 100
    
    axes[i].set_title(f"Cutoff {cutoff*100:.0f}%", fontweight='bold', fontsize=12)
    axes[i].set_xlabel(f"PC1 ({var_pc1:.1f}%)")
    axes[i].set_ylabel(f"PC2 ({var_pc2:.1f}%)")

    print(f"Random Forest {cutoff*100:.0f}%:")
    RF = RandomForestClassifier(random_state=42, n_jobs=-1)
    report_cv_rf = crossvalidation(RF, X_train_pca, y_train, 10, "f1_macro")

    print(f"XGB {cutoff*100:.0f}%:")
    XGB = XGBClassifier(n_jobs=-1)
    report_cv_xgb = crossvalidation(XGB, X_train_pca, y_train, 10, "f1_macro")

    print(f"SVM {cutoff*100:.0f}%:")
    SVM = svm.SVC(kernel="rbf", random_state=42)
    report_cv_svm = crossvalidation(SVM, X_train_pca, y_train, 10, "f1_macro")

fig.tight_layout()

fig.savefig("Griglia_PCA_Tradizionale.png", dpi=300, bbox_inches='tight') """

#plt.show()