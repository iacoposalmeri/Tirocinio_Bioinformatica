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

N_COMPONENTS = 15
K_BEST_FEATURES = 100
scenarios = ['healthy']
cutoffs = [0.03, 0.05, 0.07, 0.1, 0.15, 0.2]

results = []

for scenario in tqdm(scenarios,desc="Scenarios:"):

    x, y_binary, metadati_finali_no_disease, metadati_esclusi = caricamento_pulizia_dati("Metadati_CRC_Dataset.csv", "Abbondanze_CRC_Dataset.csv",condizione_negativa=scenario)

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

            # none
            scaler_base = StandardScaler()
            reps_train['none'] = scaler_base.fit_transform(X_tr_filt)
            reps_val['none'] = scaler_base.transform(X_val_filt)

            # SKB
            skb_raw = SelectKBest(score_func=mutual_info_classif, k=min(K_BEST_FEATURES, X_tr_filt.shape[1]))
            X_tr_skb_raw = skb_raw.fit_transform(X_tr_filt, y_tr)
            X_val_skb_raw = skb_raw.transform(X_val_filt)
            
            scaler_skb = StandardScaler()
            reps_train['SKB'] = scaler_skb.fit_transform(X_tr_skb_raw)
            reps_val['SKB'] = scaler_skb.transform(X_val_skb_raw)

            
            # Calcolo skb df
            skb_raw_columns = skb_raw.get_feature_names_out(X_tr_filt.columns)
            X_tr_skb_df = pd.DataFrame(X_tr_skb_raw, index=X_tr_filt.index, columns=skb_raw_columns)
            X_val_skb_df = pd.DataFrame(X_val_skb_raw, index=X_val_filt.index, columns=skb_raw_columns)

            # PCA
            pca = PCA(n_components=N_COMPONENTS, svd_solver='full', random_state=SEED)
            scaler_pre = StandardScaler()
            
            X_tr_scaled = scaler_pre.fit_transform(X_tr_filt)
            reps_train['PCA'] = pca.fit_transform(X_tr_scaled)
            
            X_val_scaled = scaler_pre.transform(X_val_filt)
            reps_val['PCA'] = pca.transform(X_val_scaled)

            # SKB + PCA
            pca_skb = PCA(n_components=N_COMPONENTS, svd_solver='full', random_state=SEED)
            scaler_pre_pca_skb = StandardScaler()
            
            X_tr_scaled_skb = scaler_pre_pca_skb.fit_transform(X_tr_skb_df)
            reps_train['SKB + PCA'] = pca_skb.fit_transform(X_tr_scaled_skb)
            
            X_val_scaled_skb = scaler_pre_pca_skb.transform(X_val_skb_df)
            reps_val['SKB + PCA'] = pca_skb.transform(X_val_scaled_skb)

            # Bray-Curtis
            reps_train['Bray_Curtis'] = pairwise_distances(X_tr_filt, metric='braycurtis')
            reps_val['Bray_Curtis'] = pairwise_distances(X_val_filt, X_tr_filt, metric='braycurtis')

            reps_train['SKB + Bray_Curtis'] = pairwise_distances(X_tr_skb_df, metric='braycurtis')
            reps_val['SKB + Bray_Curtis'] = pairwise_distances(X_val_skb_df, X_tr_skb_df, metric='braycurtis')


            # Calcolo CLR
            X_tr_clr_base = trasformazione_clr(X_tr_filt)
            X_val_clr_base = trasformazione_clr(X_val_filt)
            X_tr_clr_base.index = X_tr_filt.index
            X_val_clr_base.index = X_val_filt.index

            # none (with CLR)
            scaler_base_clr = StandardScaler()
            reps_train['none (with CLR)'] = scaler_base_clr.fit_transform(X_tr_clr_base)
            reps_val['none (with CLR)'] = scaler_base_clr.transform(X_val_clr_base)

            # CLR + SKB
            skb_clr = SelectKBest(score_func=mutual_info_classif, k=min(K_BEST_FEATURES, X_tr_clr_base.shape[1]))
            X_tr_clr_skb = skb_clr.fit_transform(X_tr_clr_base, y_tr)
            X_val_clr_skb = skb_clr.transform(X_val_clr_base)
            
            scaler_clr_skb = StandardScaler()
            reps_train['CLR + SKB'] = scaler_clr_skb.fit_transform(X_tr_clr_skb)
            reps_val['CLR + SKB'] = scaler_clr_skb.transform(X_val_clr_skb)

            # AITCHISON PCA
            pca_aitchison = PCA(n_components=N_COMPONENTS, svd_solver='full', random_state=SEED)
            reps_train['PCA_Aitchison'] = pca_aitchison.fit_transform(X_tr_clr_base)
            reps_val['PCA_Aitchison'] = pca_aitchison.transform(X_val_clr_base)

            # CLR + SKB + PCA
            pca_aitchison_skb = PCA(n_components=N_COMPONENTS, svd_solver='full', random_state=SEED)
            reps_train['CLR + SKB + PCA'] = pca_aitchison_skb.fit_transform(X_tr_clr_skb)
            reps_val['CLR + SKB + PCA'] = pca_aitchison_skb.transform(X_val_clr_skb)

            # RPCA
            X_tr_biom = table.Table(X_tr_filt.values.T, observation_ids=X_tr_filt.columns, sample_ids=X_tr_filt.index)
            ordination, _ = rpca(X_tr_biom, n_components=min(N_COMPONENTS, X_tr_filt.shape[1]-1))
            
            X_val_rclr_for_rpca = trasformazione_rclr_nativa(X_val_filt)
            val_rpca_projected = X_val_rclr_for_rpca @ ordination.features.values

            reps_train['RPCA'] = ordination.samples.values
            reps_val['RPCA'] = val_rpca_projected.values

            # Robust CLR
            X_tr_rclr = trasformazione_rclr_nativa(X_tr_filt)
            X_val_rclr = trasformazione_rclr_nativa(X_val_filt)
            X_tr_rclr.index = X_tr_filt.index
            X_val_rclr.index = X_val_filt.index
            
            scaler_rclr = StandardScaler()
            reps_train['Robust CLR'] = scaler_rclr.fit_transform(X_tr_rclr)
            reps_val['Robust CLR'] = scaler_rclr.transform(X_val_rclr)

            # Robust CLR + SKB
            skb_rclr = SelectKBest(score_func=mutual_info_classif, k=min(K_BEST_FEATURES, X_tr_rclr.shape[1]))
            X_tr_rclr_skb = skb_rclr.fit_transform(X_tr_rclr, y_tr)
            X_val_rclr_skb = skb_rclr.transform(X_val_rclr)
            
            scaler_rclr_skb = StandardScaler()
            reps_train['Robust CLR + SKB'] = scaler_rclr_skb.fit_transform(X_tr_rclr_skb)
            reps_val['Robust CLR + SKB'] = scaler_rclr_skb.transform(X_val_rclr_skb)
            
            # VOTAZIONE A MAGGIORANZA
            batteri_consenso = voto_maggioranza_trasformazioni(
                X_tr_filt, X_tr_clr_base, X_tr_rclr, y_tr, k=K_BEST_FEATURES
            )
            
            X_tr_consensus = X_tr_rclr[batteri_consenso]
            X_val_consensus = X_val_rclr[batteri_consenso]

            scaler_consensus = StandardScaler()
            reps_train['Consensus'] = scaler_consensus.fit_transform(X_tr_consensus)
            reps_val['Consensus'] = scaler_consensus.transform(X_val_consensus)

            # CLASSIFICATION

            models = {
                'RandomForest': RandomForestClassifier(random_state=SEED),
                'XGBoost': XGBClassifier(random_state=SEED, tree_method='hist'),
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
