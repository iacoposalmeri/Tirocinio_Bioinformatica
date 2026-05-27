""" 
THIS SCRIPT IS A BENCHMARK FOR CRC CLASSIFICATION WITH THE ADDITIONAL FOLLOWING TECHNIQUES:
- none
- SKB
- PCA
- Aitchison PCA
- PCoA with Bray-Curtis
- RPCA
- none (con CLR)
- SKB (con CLR)
- SKB + PCA
- SKB + Aitchison PCA
- SKB + PCoA with Bray-Curtis
- Robust CLR
- Robust CLR + SKB

GRID-SEARCH IS NOT USED HERE BECAUSE WE'RE TRYING TO FIND ONLY THE BEST REDUCTION TECHNIQUE
IN ANOTHER SCRIPT WE'LL USE GRID-SEARCH FOR EVERY CLASSIFICATION METHOD WITH ONLY THE BEST METHOD
"""

import warnings
warnings.filterwarnings('ignore')

import os
import random

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
from sklearn.metrics import pairwise_distances


from biom import table
from gemelli.rpca import rpca

from utils_crc2 import * 


SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
#Provare altre combinazioni con una lista?
K_VALUES = [20, 50, 100]
COMPONENTS_LIST = [15, 30, 50, 70] # Usiamo questa sia per PCA che per RPCA
scenarios = ['healthy']
cutoffs = [0.03, 0.05, 0.07, 0.1, 0.15, 0.2]



results = []

for scenario in tqdm(scenarios,desc="Scenarios:"):

    x, y_binary, metadati_finali_no_disease, metadati_esclusi = caricamento_pulizia_dati("Metadati_CRC_Dataset.csv", "Abbondanze_CRC_Dataset.csv",condizione_negativa=scenario)
    #provare a randomizzare lo split? fare 10-90?
    X_train, X_test, y_train, y_test = train_test_split(x, y_binary, test_size=0.2, random_state=SEED, stratify=y_binary)

    for cutoff in tqdm(cutoffs, desc="Cutoffs:", leave=False):

        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)

        X_train_cv = X_train.reset_index(drop=True)
        y_train_cv = y_train.reset_index(drop=True)

        for fold, (train_idx, val_idx) in enumerate(tqdm(cv.split(X_train_cv,y_train_cv), total=cv.get_n_splits(), desc="Folds progession:", leave=False)):
            
            X_tr, y_tr = X_train_cv.iloc[train_idx], y_train_cv.iloc[train_idx]
            X_val, y_val = X_train_cv.iloc[val_idx], y_train_cv.iloc[val_idx]

            bacteria_to_keep = maschera_prevalenza(X_tr,y_tr,cutoff=cutoff)
            X_tr_filt = filtraggio(X_tr, bacteria_to_keep)
            X_val_filt = filtraggio(X_val, bacteria_to_keep)

            mask_validi = (X_tr_filt != 0).any(axis=1)
            X_tr_filt = X_tr_filt[mask_validi]
            y_tr = y_tr[mask_validi]

            X_val_filt = X_val_filt[X_tr_filt.columns]

            reps_train = {}
            reps_val = {}

            # ==========================================
            # 1. TECNICHE BASE (Calcolate 1 sola volta)
            # ==========================================
            scaler_base = StandardScaler()
            reps_train['none'] = scaler_base.fit_transform(X_tr_filt)
            reps_val['none'] = scaler_base.transform(X_val_filt)


            X_tr_clr_base = trasformazione_clr(X_tr_filt)
            X_val_clr_base = trasformazione_clr(X_val_filt)
            
            scaler_base_clr = StandardScaler()
            reps_train['none (with CLR)'] = scaler_base_clr.fit_transform(X_tr_clr_base)
            reps_val['none (with CLR)'] = scaler_base_clr.transform(X_val_clr_base)

            X_tr_rclr = trasformazione_rclr_nativa(X_tr_filt)
            X_val_rclr = trasformazione_rclr_nativa(X_val_filt)
            
            scaler_rclr = StandardScaler()
            reps_train['Robust CLR'] = scaler_rclr.fit_transform(X_tr_rclr)
            reps_val['Robust CLR'] = scaler_rclr.transform(X_val_rclr)

            # ==========================================
            # 2. TECNICHE CHE DIPENDONO SOLO DA K
            # ==========================================
            for k in K_VALUES:
                k_effettivo = min(k, X_tr_filt.shape[1]) # Evita errori se k > numero di batteri rimasti
                
                # SKB Base
                skb_raw = SelectKBest(score_func=mutual_info_classif, k=k_effettivo)
                X_tr_skb_raw = skb_raw.fit_transform(X_tr_filt, y_tr)
                scaler_skb = StandardScaler()
                reps_train[f'SKB_k{k}'] = scaler_skb.fit_transform(X_tr_skb_raw)
                reps_val[f'SKB_k{k}'] = scaler_skb.transform(skb_raw.transform(X_val_filt))

                # CLR + SKB
                skb_clr = SelectKBest(score_func=mutual_info_classif, k=k_effettivo)
                X_tr_clr_skb = skb_clr.fit_transform(X_tr_clr_base, y_tr)
                scaler_clr_skb = StandardScaler()
                reps_train[f'CLR_SKB_k{k}'] = scaler_clr_skb.fit_transform(X_tr_clr_skb)
                reps_val[f'CLR_SKB_k{k}'] = scaler_clr_skb.transform(skb_clr.transform(X_val_clr_base))

            # ==========================================
            # 3. TECNICHE CHE DIPENDONO SOLO DAI COMPONENTI (PCA)
            # ==========================================
            for c in COMPONENTS_LIST:
                # RETE DI SICUREZZA: Prende il numero minimo tra le componenti richieste, 
                # il numero di pazienti (-1) e il numero di batteri (-1)
                c_effettive = min(c, X_tr_filt.shape[0] - 1, X_tr_filt.shape[1] - 1)
                
                # PCA Base
                pca = PCA(n_components=c_effettive, svd_solver='full', random_state=SEED)
                X_tr_scaled = StandardScaler().fit_transform(X_tr_filt)
                reps_train[f'PCA_c{c}'] = pca.fit_transform(X_tr_scaled)
                reps_val[f'PCA_c{c}'] = pca.transform(StandardScaler().fit(X_tr_filt).transform(X_val_filt))

                # Aitchison PCA
                pca_aitchison = PCA(n_components=c_effettive, svd_solver='full', random_state=SEED)
                reps_train[f'PCA_Aitchison_c{c}'] = pca_aitchison.fit_transform(X_tr_clr_base)
                reps_val[f'PCA_Aitchison_c{c}'] = pca_aitchison.transform(X_val_clr_base)

            # ==========================================
            # 4. TECNICHE COMBINATE (K + Componenti)
            # ==========================================
            for k in K_VALUES:
                k_effettivo = min(k, X_tr_filt.shape[1])
                
                skb_raw = SelectKBest(score_func=mutual_info_classif, k=k_effettivo)
                X_tr_skb_raw = skb_raw.fit_transform(X_tr_filt, y_tr)
                
                skb_clr = SelectKBest(score_func=mutual_info_classif, k=k_effettivo)
                X_tr_clr_skb = skb_clr.fit_transform(X_tr_clr_base, y_tr)

                for c in COMPONENTS_LIST:
                    # Rete di sicurezza per la PCA dopo il SelectKBest
                    c_effettive_skb = min(c, X_tr_skb_raw.shape[0] - 1, X_tr_skb_raw.shape[1] - 1)
                    
                    # SKB + PCA
                    pca_skb = PCA(n_components=c_effettive_skb, svd_solver='full', random_state=SEED)
                    X_tr_scaled_skb = StandardScaler().fit_transform(X_tr_skb_raw)
                    reps_train[f'SKB_k{k}_PCA_c{c}'] = pca_skb.fit_transform(X_tr_scaled_skb)
                    reps_val[f'SKB_k{k}_PCA_c{c}'] = pca_skb.transform(StandardScaler().fit(X_tr_skb_raw).transform(skb_raw.transform(X_val_filt)))

                    # CLR + SKB + PCA
                    pca_aitchison_skb = PCA(n_components=c_effettive_skb, svd_solver='full', random_state=SEED)
                    reps_train[f'CLR_SKB_k{k}_PCA_c{c}'] = pca_aitchison_skb.fit_transform(X_tr_clr_skb)
                    reps_val[f'CLR_SKB_k{k}_PCA_c{c}'] = pca_aitchison_skb.transform(skb_clr.transform(X_val_clr_base))

            # ==========================================
            # 5. RPCA (Grid Search)
            # ==========================================
            X_tr_biom = table.Table(X_tr_filt.values.T, observation_ids=X_tr_filt.columns, sample_ids=X_tr_filt.index)
            X_val_rclr_for_rpca = trasformazione_rclr_nativa(X_val_filt)

            for c in COMPONENTS_LIST:
                c_effettive_rpca = min(c, X_tr_filt.shape[0] - 1, X_tr_filt.shape[1] - 1)
                
                ordination, _ = rpca(X_tr_biom, n_components=c_effettive_rpca)
                reps_train[f'RPCA_c{c}'] = ordination.samples.values
                reps_val[f'RPCA_c{c}'] = (X_val_rclr_for_rpca @ ordination.features.values).values
            models = {
                'RandomForest': RandomForestClassifier(random_state=SEED, n_jobs=-1),
                'XGBoost': XGBClassifier(random_state=SEED, tree_method='hist', n_jobs=-1),
                'SVM': SVC(kernel="rbf", probability=True, random_state=SEED, cache_size=2000)
            }

            for rep_name, X_train_final in reps_train.items():
                if rep_name not in reps_val:
                    continue
                    
                X_val_final = reps_val[rep_name]

                if X_train_final.shape[0] != len(y_tr):
                    print(f"ERRORE: Disallineamento trovato!")
                    print(f"Tecnica: {rep_name}")
                    print(f"X_train_final shape: {X_train_final.shape}")
                    print(f"y_tr length: {len(y_tr)}")

                    continue
                
                for model_name, clf in models.items():
                    clf.fit(X_train_final, y_tr)
                    
                    y_pred = clf.predict(X_val_final)
                    y_proba = clf.predict_proba(X_val_final)[:, 1]
                    
                    mcc = float(matthews_corrcoef(y_val, y_pred))
                    f1 = float(f1_score(y_val, y_pred))
                    auc = float(roc_auc_score(y_val, y_proba))

                    results.append({
                        'Scenario': scenario,
                        'Cutoff': cutoff,
                        'Technique': rep_name,
                        'Model': model_name,
                        'Fold': fold + 1,
                        'MCC': mcc,
                        'F1': f1,
                        'AUC': auc,
                    })

        pd.DataFrame(results).to_csv("Backup_Temp_Benchmark.csv", index=False)


df_raw = pd.DataFrame(results)
df_final = df_raw.groupby(['Scenario', 'Cutoff', 'Technique', 'Model']).agg({
    'MCC': 'mean', 'F1': ['mean', 'std'], 'AUC': 'mean',
}).reset_index()

df_final.columns = ['_'.join(col).strip() if col[1] else col[0] for col in df_final.columns.values]
df_final.to_csv("Fase1_Benchmark_Riduzioni.csv", index=False)
