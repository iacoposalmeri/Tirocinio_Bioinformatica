""" 
THIS SCRIPT IS A BENCHMARK FOR CRC CLASSIFICATION.
EVALUATING COMBINATIONS OF:
- Transformations: None (Raw), CLR, Robust CLR (rCLR)
- Feature Selection: None, SelectKBest (varying k)
- Dimensionality Reduction: None, PCA (varying explained variance)

NOTE: Standard PCA is explicitly skipped for rCLR transformed data.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import matthews_corrcoef, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# Import custom utils
from utils_crc2 import * 
from sklearnex import patch_sklearn
patch_sklearn()

# --- PARAMETRI GLOBALI ---
SEED = 42
K_BEST_VALUES = [30, 50, 100, 150, 200, 250, 300]
PCA_VARIANCES = [0.70, 0.80, 0.90, 0.95]
SCENARIOS = ['healthy']
CUTOFFS = [0.03, 0.05, 0.07, 0.1, 0.15, 0.2]

results = []

for scenario in tqdm(SCENARIOS, desc="Scenarios:"):

    x, y_binary, metadati_finali_no_disease, metadati_esclusi = caricamento_pulizia_dati("Metadati_CRC_Dataset.csv", "Abbondanze_CRC_Dataset.csv", condizione_negativa=scenario)

    X_train, X_test, y_train, y_test = train_test_split(x, y_binary, test_size=0.2, random_state=SEED, stratify=y_binary)

    for cutoff in tqdm(CUTOFFS, desc="Cutoffs:", leave=False):

        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)

        X_train_cv = X_train.reset_index(drop=True)
        y_train_cv = y_train.reset_index(drop=True)

        for fold, (train_idx, val_idx) in enumerate(tqdm(cv.split(X_train_cv, y_train_cv), total=cv.get_n_splits(), desc="Folds progression:", leave=False)):
            
            X_tr, y_tr = X_train_cv.iloc[train_idx], y_train_cv.iloc[train_idx]
            X_val, y_val = X_train_cv.iloc[val_idx], y_train_cv.iloc[val_idx]

            # Filtraggio prevalenza
            bacteria_to_keep = maschera_prevalenza(X_tr, y_tr, cutoff=cutoff)
            X_tr_filt = filtraggio(X_tr, bacteria_to_keep)
            X_val_filt = filtraggio(X_val, bacteria_to_keep)

            mask_validi = (X_tr_filt != 0).any(axis=1)
            X_tr_filt = X_tr_filt[mask_validi]
            y_tr = y_tr[mask_validi]
            X_val_filt = X_val_filt[X_tr_filt.columns]

            # --- PREPARAZIONE DELLE TRASFORMAZIONI DI BASE ---
            # È fondamentale trasformare il dato PRIMA di selezionare le feature 
            # per non alterare la media geometrica intrinseca nella CLR.
            transformed_data = {}
            
            # 1. Nessuna trasformazione
            transformed_data['Raw'] = (X_tr_filt.copy(), X_val_filt.copy())
            
            # 2. CLR
            X_tr_clr = trasformazione_clr(X_tr_filt)
            X_val_clr = trasformazione_clr(X_val_filt)
            X_tr_clr.index, X_val_clr.index = X_tr_filt.index, X_val_filt.index
            transformed_data['CLR'] = (X_tr_clr, X_val_clr)
            
            # 3. Robust CLR
            X_tr_rclr = trasformazione_rclr_nativa(X_tr_filt)
            X_val_rclr = trasformazione_rclr_nativa(X_val_filt)
            X_tr_rclr.index, X_val_rclr.index = X_tr_filt.index, X_val_filt.index
            transformed_data['rCLR'] = (X_tr_rclr, X_val_rclr)

            reps_train = {}
            reps_val = {}

            # --- GENERAZIONE COMBINAZIONI DINAMICHE ---
            for trans_name, (X_tr_t, X_val_t) in transformed_data.items():
                
                # --- A. SOLO TRASFORMAZIONE (Nessuna riduzione/selezione) ---
                scaler_base = StandardScaler()
                reps_train[f'{trans_name}_AllFeatures'] = scaler_base.fit_transform(X_tr_t)
                reps_val[f'{trans_name}_AllFeatures'] = scaler_base.transform(X_val_t)

                # --- B. SOLO PCA (Senza SelectKBest) ---
                if trans_name != 'rCLR':  # Escludiamo PCA se usiamo rCLR
                    for var in PCA_VARIANCES:
                        pca = PCA(n_components=var, svd_solver='full', random_state=SEED)
                        # Applico PCA sui dati già scalati
                        X_tr_pca = pca.fit_transform(reps_train[f'{trans_name}_AllFeatures'])
                        X_val_pca = pca.transform(reps_val[f'{trans_name}_AllFeatures'])
                        
                        scaler_pca = StandardScaler()
                        reps_train[f'{trans_name}_AllFeatures_PCA_{int(var*100)}'] = scaler_pca.fit_transform(X_tr_pca)
                        reps_val[f'{trans_name}_AllFeatures_PCA_{int(var*100)}'] = scaler_pca.transform(X_val_pca)

                # --- C. SELECT K BEST E SUE COMBINAZIONI ---
                for k in K_BEST_VALUES:
                    # Continuiamo a usare eff_k per non far crashare scikit-learn
                    eff_k = min(k, X_tr_t.shape[1])
                    
                    skb = SelectKBest(score_func=mutual_info_classif, k=eff_k)
                    X_tr_skb = skb.fit_transform(X_tr_t, y_tr)
                    X_val_skb = skb.transform(X_val_t)

                    scaler_skb = StandardScaler()
                    X_tr_skb_scaled = scaler_skb.fit_transform(X_tr_skb)
                    X_val_skb_scaled = scaler_skb.transform(X_val_skb)

                    # SOLO SKB -> USIAMO 'k' E NON 'eff_k' PER IL NOME
                    reps_train[f'{trans_name}_SKB_{k}'] = X_tr_skb_scaled
                    reps_val[f'{trans_name}_SKB_{k}'] = X_val_skb_scaled

                    # SKB + PCA
                    if trans_name != 'rCLR':
                        for var in PCA_VARIANCES:
                            pca_skb = PCA(n_components=var, svd_solver='full', random_state=SEED)
                            X_tr_skb_pca = pca_skb.fit_transform(X_tr_skb_scaled)
                            X_val_skb_pca = pca_skb.transform(X_val_skb_scaled)

                            scaler_skb_pca = StandardScaler()
                            # USIAMO 'k' E NON 'eff_k' PER IL NOME
                            reps_train[f'{trans_name}_SKB_{k}_PCA_{int(var*100)}'] = scaler_skb_pca.fit_transform(X_tr_skb_pca)
                            reps_val[f'{trans_name}_SKB_{k}_PCA_{int(var*100)}'] = scaler_skb_pca.transform(X_val_skb_pca)

            # --- CLASSIFICATION ---
            # Modelli con parametri di default + seed
            models = {
                'RandomForest': RandomForestClassifier(random_state=SEED, n_jobs=-1),
                'XGBoost': XGBClassifier(random_state=SEED, n_jobs=-1),
                'SVM': SVC(probability=True, random_state=SEED)
            }

            for rep_name, X_train_final in reps_train.items():
                if rep_name not in reps_val:
                    continue
                    
                X_val_final = reps_val[rep_name]

                # Sanity check dimensionale
                if X_train_final.shape[0] != len(y_tr):
                    print(f"ERRORE: Disallineamento trovato! Tecnica: {rep_name}")
                    continue
                
                for model_name, clf in models.items():
                    clf.fit(X_train_final, y_tr)
                    
                    y_pred = clf.predict(X_val_final)
                    y_proba = clf.predict_proba(X_val_final)[:, 1]
                    
                    results.append({
                        'Scenario': scenario,
                        'Cutoff': cutoff,
                        'Technique': rep_name,
                        'Model': model_name,
                        'Fold': fold + 1,
                        'MCC': matthews_corrcoef(y_val, y_pred),
                        'F1': f1_score(y_val, y_pred),
                        'AUC': roc_auc_score(y_val, y_proba),
                    })


# Estrazione e salvataggio dei risultati
df_raw = pd.DataFrame(results)

# Aggiunta di colonne specifiche per facilitare le analisi future (Opzionale ma molto utile)
df_raw['Base_Transform'] = df_raw['Technique'].apply(lambda x: x.split('_')[0])
df_raw['Has_PCA'] = df_raw['Technique'].apply(lambda x: 'PCA' in x)
df_raw['Has_SKB'] = df_raw['Technique'].apply(lambda x: 'SKB' in x)

df_final = df_raw.groupby(['Scenario', 'Cutoff', 'Technique', 'Base_Transform', 'Has_PCA', 'Has_SKB', 'Model']).agg({
    'MCC': 'mean', 'F1': ['mean', 'std'], 'AUC': 'mean',
}).reset_index()

# Appiattimento MultiIndex delle colonne generato dall'agg
df_final.columns = ['_'.join(col).strip() if col[1] else col[0] for col in df_final.columns.values]
df_final.to_csv("Fase1_Benchmark_Riduzioni_gemini.csv", index=False)