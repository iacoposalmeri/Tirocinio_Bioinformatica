import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from xgboost import XGBClassifier
from funzioni_crc import maschera_prevalenza, caricamento_pulizia_dati, filtraggio, trasformazione_clr, standard_scaler, crossvalidation, pca_classica, pca_grafico, pca_grafico2
from sklearn.feature_selection import SelectKBest, mutual_info_classif
cutoffs = [0.03, 0.05, 0.07, 0.1, 0.15, 0.2]
varianze_pca = [0.7, 0.8, 0.9, 0.95] #varianze da testare per la PCA

x, y_binary, metadati_finali_no_desease, metadati_esclusi = caricamento_pulizia_dati("Metadati_CRC_Dataset.csv", "Abbondanze_CRC_Dataset.csv")

X_train, X_test, y_train, y_test = train_test_split(x, y_binary, test_size=0.2, random_state=42, stratify=y_binary)

df_differenze = {"cutoff": [], "feature_rimaste": []}

for cutoff in cutoffs:
        batteri_da_tenere = maschera_prevalenza(X_train, y_train, cutoff)
        
        print(f"Prima del filtraggio al {cutoff*100:.0f}%: {X_train.shape}")
        X_train_filtrato = filtraggio(X_train, batteri_da_tenere)
        X_test_filtrato = filtraggio(X_test, batteri_da_tenere)
        print(f"Dopo il filtraggio al {cutoff*100:.0f}%: {X_train_filtrato.shape}")
        X_train_filtrato.T.index.to_series().to_csv(f"feature_rimaste_cutoff_{int(cutoff*100)}%.csv", index=False) 
        df_differenze["cutoff"].append(cutoff)
        df_differenze["feature_rimaste"].append(X_train_filtrato.shape[1])
        

df_differenze = pd.DataFrame(df_differenze)
df_differenze.to_csv("differenze_cutoff.csv", index=False)