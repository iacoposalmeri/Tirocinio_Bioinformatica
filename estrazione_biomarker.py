import pandas as pd
from xgboost import XGBClassifier
from sklearn.feature_selection import RFE
from utils_crc2 import caricamento_pulizia_dati, maschera_prevalenza, filtraggio, trasformazione_rclr_nativa

# 1. Caricamento e preparazione
X, y_binary, _, _ = caricamento_pulizia_dati("Metadati_CRC_Dataset.csv", "Abbondanze_CRC_Dataset.csv", condizione_negativa="healthy")

# 2. Cutoff severo al 20% (come il modello vincente)
batteri_da_tenere = maschera_prevalenza(X, y_binary, cutoff=0.20)
X_filt = filtraggio(X, batteri_da_tenere)

# Applichiamo la trasformazione nativa RCLR (lo spazio dove hai stravinto)
X_rclr = trasformazione_rclr_nativa(X_filt)

# 3. Impostiamo il Motore RFE (con i parametri vincitori della Grid Search)
motore_base = XGBClassifier(
    learning_rate=0.01,
    max_depth=3,
    min_child_weight=1,
    n_estimators=300,
    subsample=0.8,
    random_state=42,
    tree_method='hist'
)

# 4. Applichiamo la RFE per trovare i magici 50
selettore_rfe = RFE(estimator=motore_base, n_features_to_select=50, step=0.1)
selettore_rfe.fit(X_rclr, y_binary)

# 5. Estrazione dei nomi
nomi_batteri_rclr = X_rclr.columns
indici_vincenti = selettore_rfe.get_support(indices=True)
batteri_50 = nomi_batteri_rclr[indici_vincenti]

print(f"I 50 BIOMARCATORI DEFINITIVI PER IL CANCRO COLORETTALE:\n")
for i, batterio in enumerate(batteri_50, 1):
    print(f"{i}. {batterio}")