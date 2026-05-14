import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import svm
from sklearn.metrics import classification_report
from funzioni_crc import maschera_prevalenza, caricamento_pulizia_dati, filtraggio, applica_algoritmo, trasformazione_clr, standard_scaler, pca_classica

# ==========================================
# 1. SETUP DELLA PIPELINE VINCITRICE
# ==========================================
cutoff = 0.05
varianza_pca = 0.95 

print("Caricamento dati...")
x, y_binary, metadati_finali_no_desease, metadati_esclusi = caricamento_pulizia_dati("Metadati_CRC_Dataset.csv", "Abbondanze_CRC_Dataset.csv")

# Split e CLR
X_train, X_test, y_train, y_test = train_test_split(x, y_binary, test_size=0.2, random_state=42, stratify=y_binary)
X_train_clr = trasformazione_clr(X_train)
X_test_clr = trasformazione_clr(X_test)

# Filtraggio Prevalenza
batteri_da_tenere = maschera_prevalenza(X_train, y_train, cutoff)
print(f"Prima del filtraggio al {cutoff*100:.0f}%: {X_train_clr.shape}")

X_train_filtrato = filtraggio(X_train_clr, batteri_da_tenere)
X_test_filtrato = filtraggio(X_test_clr, batteri_da_tenere)
print(f"Dopo il filtraggio al {cutoff*100:.0f}%: {X_train_filtrato.shape}")

# Scaling 
X_train_scaled, X_test_scaled = standard_scaler(X_train_filtrato, X_test_filtrato)

# PCA di Aitchison
print(f"Esecuzione PCA ({varianza_pca*100:.0f}% varianza)...")
X_train_pca, X_test_pca, pca_fitted = pca_classica(X_train_scaled, X_test_scaled, n_components=varianza_pca)
num_componenti = pca_fitted.n_components_
print(f"La PCA ha estratto {num_componenti} componenti principali.")

svm_finale = svm.SVC(random_state=42, C=1, gamma='scale', kernel='rbf')
report, cm = applica_algoritmo(svm_finale, X_train_pca, X_test_pca, y_train, y_test, "SVM_finale")