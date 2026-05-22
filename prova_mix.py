import pandas as pd
import numpy as np  # Aggiunto per il calcolo di media e deviazione standard a fine ciclo
from sklearn.model_selection import train_test_split, StratifiedKFold  # Aggiunto StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import f1_score  # Importato per calcolare l'F1-score manualmente sui fold
from funzioni_crc import maschera_prevalenza, caricamento_pulizia_dati, filtraggio, trasformazione_clr, standard_scaler

cutoffs = [0.03, 0.05, 0.07]
k_skb = [100]

# Caricamento dati originale
x, y_binary, metadati_finali_no_desease, metadati_esclusi = caricamento_pulizia_dati("Metadati_CRC_Dataset.csv", "Abbondanze_CRC_Dataset.csv")

# Train/Test split iniziale
X_train, X_test, y_train, y_test = train_test_split(x, y_binary, test_size=0.2, random_state=42, stratify=y_binary)

risultati_cv = {"cutoff": [], "k_skb": [], "Modello": [], "Media_f1_macro": [], "Deviazione_standard_f1_macro": []}

# Resettiamo gli indici per evitare disallineamenti con .iloc durante la Cross-Validation (best practice da final_benchmark)
X_train_cv = X_train.reset_index(drop=True)
y_train_cv = y_train.reset_index(drop=True)

for cutoff in cutoffs:
    for ks in k_skb:
        # Inizializziamo la StratifiedKFold a 10 fold (come nel tuo cross_val_score originario)
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        
        # Liste temporanee per memorizzare i punteggi f1_macro ottenuti in ognuno dei 10 fold
        scores_rf = []
        scores_xgb = []
        scores_svm = []
        
        print(f"Esecuzione 10-Fold CV per Cutoff: {cutoff*100:.0f}%, K_SKB: {ks}")
        
        # Ciclo manuale sui fold (Logica estratta dal benchmark del tuo collega)
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_cv, y_train_cv)):
            # Split dei dati del fold corrente in Train e Validation
            X_tr, y_tr = X_train_cv.iloc[train_idx], y_train_cv.iloc[train_idx]
            X_val, y_val = X_train_cv.iloc[val_idx], y_train_cv.iloc[val_idx]
            
            # 1. Calcolo della maschera di prevalenza SOLO sul set di addestramento del fold (X_tr)
            batteri_da_tenere = maschera_prevalenza(X_tr, y_tr, cutoff)
            
            # 2. Trasformazione CLR applicata in modo indipendente (riga per riga)
            #X_tr_clr = trasformazione_clr(X_tr)
            #X_val_clr = trasformazione_clr(X_val)
            
            # 3. Filtraggio basato sulle sole feature ammesse dalla maschera del fold
            X_tr_filtrato = filtraggio(X_tr, batteri_da_tenere)
            X_val_filtrato = filtraggio(X_val, batteri_da_tenere)
            
            # Sicurezza: impedisce a SelectKBest di chiedere più feature di quelle sopravvissute al filtro di prevalenza
            ks_effettivo = min(ks, X_tr_filtrato.shape[1])
            SKB = SelectKBest(mutual_info_classif, k=ks_effettivo)
            
            # 4. Feature Selection: impariamo (fit) le feature migliori da X_tr e trasformiamo entrambi
            X_tr_skb = SKB.fit_transform(X_tr_filtrato, y_tr)
            X_val_skb = SKB.transform(X_val_filtrato)
            
            nomi_feature = [X_tr_filtrato.columns[i] for i in SKB.get_support(indices=True)]
            X_tr_skb_df = pd.DataFrame(X_tr_skb, columns=nomi_feature, index=X_tr.index)
            X_val_skb_df = pd.DataFrame(X_val_skb, columns=nomi_feature, index=X_val.index)   
            
            # 5. Standard Scaling: fittiamo il media/varianza su X_tr e trasformiamo X_val
            X_tr_scaled, X_val_scaled = standard_scaler(X_tr_skb_df, X_val_skb_df)
            
            # --- Fase di Fit e Predict per questo specifico fold ---
            
            # Random Forest (sui dati non scalati, come nel tuo script originario)
            RF = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')  # Aggiunta di class_weight per bilanciare le classi
            RF.fit(X_tr_skb_df, y_tr)
            preds_rf = RF.predict(X_val_skb_df)
            scores_rf.append(f1_score(y_val, preds_rf, average='macro'))

            # XGBoost (sui dati non scalati)
            XGB = XGBClassifier(n_jobs=-1, random_state=42)
            XGB.fit(X_tr_skb_df, y_tr)
            preds_xgb = XGB.predict(X_val_skb_df)
            scores_xgb.append(f1_score(y_val, preds_xgb, average='macro'))

            # SVM (sui dati scalati con standard_scaler)
            SVM = svm.SVC(kernel="rbf", random_state=42, class_weight='balanced')  # Aggiunta di class_weight per bilanciare le classi
            SVM.fit(X_tr_scaled, y_tr)
            preds_svm = SVM.predict(X_val_scaled)
            scores_svm.append(f1_score(y_val, preds_svm, average='macro'))
        
        # Una volta terminati i 10 fold per questa configurazione, calcoliamo metriche riassuntive
        # Random Forest
        risultati_cv["cutoff"].append(cutoff)
        risultati_cv["k_skb"].append(ks)
        risultati_cv["Modello"].append("Random Forest")
        risultati_cv["Media_f1_macro"].append(np.mean(scores_rf))
        risultati_cv["Deviazione_standard_f1_macro"].append(np.std(scores_rf))

        # XGB
        risultati_cv["cutoff"].append(cutoff)
        risultati_cv["k_skb"].append(ks)
        risultati_cv["Modello"].append("XGB")
        risultati_cv["Media_f1_macro"].append(np.mean(scores_xgb))
        risultati_cv["Deviazione_standard_f1_macro"].append(np.std(scores_xgb))

        # SVM
        risultati_cv["cutoff"].append(cutoff)
        risultati_cv["k_skb"].append(ks)
        risultati_cv["Modello"].append("SVM")
        risultati_cv["Media_f1_macro"].append(np.mean(scores_svm))
        risultati_cv["Deviazione_standard_f1_macro"].append(np.std(scores_svm))

# Esportazione finale dei risultati puliti dal leakage
df_risultati_cv = pd.DataFrame(risultati_cv)
df_risultati_cv.to_csv("risultati_crossvalidation_skb_mix.csv", index=False)
print("\nCOMPLETATO! I risultati non distorti sono stati salvati in risultati_crossvalidation_skb_mix.csv")