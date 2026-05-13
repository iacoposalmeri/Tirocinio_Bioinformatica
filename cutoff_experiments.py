from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from xgboost import XGBClassifier
from utils_crc import *
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

cutoffs = [0.03, 0.05, 0.07, 0.1, 0.15, 0.2]

# aggiungere il parametro <<condizione_negativa = 'healthy'>> per il confronto fra pazienti malati di CRC e persone completamente sane
x, y_binary, metadati_finali_no_desease, metadati_esclusi = caricamento_pulizia_dati("Metadati_CRC_Dataset.csv", "Abbondanze_CRC_Dataset.csv")


X_train, X_test, y_train, y_test = train_test_split(x, y_binary, test_size=0.2, random_state=42, stratify=y_binary)

# PCA TRADIZIONALE

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
axes = axes.flatten()

for i, cutoff in enumerate(cutoffs):

    batteri_da_tenere = maschera_prevalenza(X_train, y_train, cutoff)

    print(f"Prima del filtraggio al {cutoff*100:.0f}%: {X_train.shape}")
    X_train_filtrato = filtraggio(X_train, batteri_da_tenere)
    X_test_filtrato = filtraggio(X_test, batteri_da_tenere)
    X_train_scaled, X_test_scaled = standard_scaler(X_train_filtrato, X_test_filtrato)
    print(f"Dopo il filtraggio al {cutoff*100:.0f}%: {X_train_filtrato.shape}")


    #BLOCCO PCA
    pca = PCA(n_components=100,random_state=42)

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

fig.savefig("Griglia_PCA_Tradizionale.png", dpi=300, bbox_inches='tight')

# AITCHISON PCA

fig_2, axes_2 = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
axes_2 = axes_2.flatten()

for i,cutoff in enumerate(cutoffs):

    batteri_da_tenere = maschera_prevalenza(X_train, y_train, cutoff)

    """ print(f"Prima del filtraggio al {cutoff*100:.0f}%: {X_train.shape}")
    X_train_filtrato = filtraggio(X_train, batteri_da_tenere)
    X_test_filtrato = filtraggio(X_test, batteri_da_tenere)
    print(f"Dopo il filtraggio al {cutoff*100:.0f}%: {X_train_filtrato.shape}")
    X_train_clr = trasformazione_clr(X_train_filtrato)
    X_test_clr = trasformazione_clr(X_test_filtrato) 
    
    X_train_scaled, X_test_scaled = standard_scaler(X_train_clr, X_test_clr)"""

    X_train_clr = trasformazione_clr(X_train)
    X_test_clr = trasformazione_clr(X_test)
    
    X_train_filtrato = filtraggio(X_train_clr, batteri_da_tenere)
    X_test_filtrato = filtraggio(X_test_clr, batteri_da_tenere)

    X_train_scaled, X_test_scaled = standard_scaler(X_train_filtrato, X_test_filtrato)

    

    #BLOCCO PCA
    pca = PCA(n_components=100,random_state=42)

    X_train_pca = pca.fit_transform(X_train_scaled,y_train)
    X_test_pca = pca.transform(X_test_scaled)

    varianza_totale = pca.explained_variance_ratio_.sum() * 100
    print(f"La PCA ha compresso i dati in {pca.n_components_} dimensioni salvando il {varianza_totale:.1f}% dell'informazione.")

    sns.scatterplot(
        x = X_train_pca[:, 0],
        y = X_train_pca[:, 1],
        hue = y_train.map({0: 'Control', 1: 'CRC'}),
        palette = {'Control' : "#EE752A", 'CRC' : "#02CAF1"},
        ax = axes_2[i],
        alpha = 0.7,
        s = 50
    )

    var_pc1 = pca.explained_variance_ratio_[0] * 100
    var_pc2 = pca.explained_variance_ratio_[1] * 100
    
    axes_2[i].set_title(f"Cutoff {cutoff*100:.0f}%", fontweight='bold', fontsize=12)
    axes_2[i].set_xlabel(f"PC1 ({var_pc1:.1f}%)")
    axes_2[i].set_ylabel(f"PC2 ({var_pc2:.1f}%)")

    print(f"Random Forest {cutoff*100:.0f}%:")
    RF = RandomForestClassifier(random_state=42, n_jobs=-1)
    report_cv_rf = crossvalidation(RF, X_train_pca, y_train, 10, "f1_macro")

    print(f"XGB {cutoff*100:.0f}%:")
    XGB = XGBClassifier(n_jobs=-1)
    report_cv_xgb = crossvalidation(XGB, X_train_pca, y_train, 10, "f1_macro")

    print(f"SVM {cutoff*100:.0f}%:")
    SVM = svm.SVC(kernel="rbf", random_state=42)
    report_cv_svm = crossvalidation(SVM, X_train_pca, y_train, 10, "f1_macro")

fig_2.tight_layout()

fig_2.savefig("Griglia_PCA_Aitchison.png", dpi=300, bbox_inches='tight')