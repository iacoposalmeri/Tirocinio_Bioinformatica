""" 
THIS SCRIPT IS A BENCHMARK FOR CRC CLASSIFICATION WITH THE ADDITIONAL FOLLOWING TECHNIQUES:
- none (con CLR)
- SKB (con CLR)
- SKB + PCA
- SKB + Aitchison PCA
- SKB + PCoA with Bray-Curtis
- Robust CLR (senza RPCA)
- SKB + Robust CLR

GRID-SEARCH IS NOT USED HERE BECAUSE WE'RE TRYING TO FIND ONLY THE BEST REDUCTION TECHNIQUE
IN ANOTHER SCRIPT WE'LL USE GRID-SEARCH FOR EVERY CLASSIFICATION METHOD WITH ONLY THE BEST METHOD
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
from sklearn.metrics import pairwise_distances


from biom import table
from gemelli.rpca import rpca

from utils_crc2 import * 


from sklearnex import patch_sklearn
patch_sklearn()


SEED = 42
N_COMPONENTS = 15
K_BEST_FEATURES = 100
scenarios = ['healthy']
cutoffs = [0.03, 0.05, 0.07]

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

            # --- 1. SKB preliminare ---
            skb = SelectKBest(score_func=mutual_info_classif, k=min(K_BEST_FEATURES, X_tr_filt.shape[1]))
            X_tr_skb_raw = skb.fit_transform(X_tr_filt, y_tr)
            X_val_skb_raw = skb.transform(X_val_filt)

            skb_columns = skb.get_feature_names_out(X_tr_filt.columns)

            X_tr_skb_df = pd.DataFrame(X_tr_skb_raw, index=X_tr_filt.index, columns=skb_columns)
            X_val_skb_df = pd.DataFrame(X_val_skb_raw, index=X_val_filt.index, columns=skb_columns)

            # --- 2. none (con CLR) --- CORRETTO
            X_tr_clr_base = trasformazione_clr(X_tr_filt)
            X_val_clr_base = trasformazione_clr(X_val_filt)
            
            X_tr_clr_base.index = X_tr_filt.index
            X_val_clr_base.index = X_val_filt.index

            scaler_base_clr = StandardScaler()
            reps_train['none (with CLR)'] = scaler_base_clr.fit_transform(X_tr_clr_base)
            reps_val['none (with CLR)'] = scaler_base_clr.transform(X_val_clr_base)

            # --- 3. SKB (con CLR) --- NUOVO
            X_tr_skb_clr = trasformazione_clr(X_tr_skb_df)
            X_val_skb_clr = trasformazione_clr(X_val_skb_df)
            X_tr_skb_clr.index = X_tr_skb_df.index
            X_val_skb_clr.index = X_val_skb_df.index
            scaler_skb_clr = StandardScaler()
            reps_train['SKB (with CLR)'] = scaler_skb_clr.fit_transform(X_tr_skb_clr)
            reps_val['SKB (with CLR)'] = scaler_skb_clr.transform(X_val_skb_clr)

            # --- 4. SKB + PCA ---
            pca = PCA(n_components=N_COMPONENTS, svd_solver='full', random_state=SEED)
            scaler_pre_pca = StandardScaler()
            scaler_post_pca = StandardScaler()

            X_tr_scaled = scaler_pre_pca.fit_transform(X_tr_skb_df)
            X_tr_pca = pca.fit_transform(X_tr_scaled)
            reps_train['SKB + PCA'] = scaler_post_pca.fit_transform(X_tr_pca)

            X_val_scaled = scaler_pre_pca.transform(X_val_skb_df)
            X_val_pca = pca.transform(X_val_scaled)
            reps_val['SKB + PCA'] = scaler_post_pca.transform(X_val_pca)

            # --- 5. SKB + AITCHISON PCA ---
            pca_aitchison = PCA(n_components=N_COMPONENTS, svd_solver='full', random_state=SEED)
            scaler_pca_aitchison = StandardScaler()

            X_tr_pca_aitchison = pca_aitchison.fit_transform(X_tr_skb_clr) # Uso l'SKB già convertito in CLR sopra
            reps_train['SKB + PCA_Aitchison'] = scaler_pca_aitchison.fit_transform(X_tr_pca_aitchison)

            X_val_pca_aitchison = pca_aitchison.transform(X_val_skb_clr)
            reps_val['SKB + PCA_Aitchison'] = scaler_pca_aitchison.transform(X_val_pca_aitchison)

            # --- 6. SKB + PCoA (Bray-Curtis) --- CORRETTO
            reps_train['SKB + Bray_Curtis'] = pairwise_distances(X_tr_skb_df, metric='braycurtis')
            reps_val['SKB + Bray_Curtis'] = pairwise_distances(X_val_skb_df, X_tr_skb_df, metric='braycurtis')

            # --- 7. Robust CLR (senza RPCA) --- NUOVO
            X_tr_rclr = trasformazione_rclr_nativa(X_tr_filt)
            X_val_rclr = trasformazione_rclr_nativa(X_val_filt)
            X_tr_rclr.index = X_tr_filt.index
            X_val_rclr.index = X_val_filt.index
            scaler_rclr = StandardScaler()
            reps_train['Robust CLR'] = scaler_rclr.fit_transform(X_tr_rclr)
            reps_val['Robust CLR'] = scaler_rclr.transform(X_val_rclr)

            # --- 8. SKB + Robust CLR --- NUOVO
            X_tr_skb_rclr = trasformazione_rclr_nativa(X_tr_skb_df)
            X_val_skb_rclr = trasformazione_rclr_nativa(X_val_skb_df)
            X_tr_skb_rclr.index = X_tr_skb_df.index
            X_val_skb_rclr.index = X_val_skb_df.index
            scaler_skb_rclr = StandardScaler()
            reps_train['SKB + Robust CLR'] = scaler_skb_rclr.fit_transform(X_tr_skb_rclr)
            reps_val['SKB + Robust CLR'] = scaler_skb_rclr.transform(X_val_skb_rclr)

            
            
            # CLASSIFICATION

            models = {
                'RandomForest': RandomForestClassifier(n_jobs=-1, random_state=SEED, class_weight='balanced'),
                'XGBoost': XGBClassifier(random_state=SEED, n_jobs=-1, tree_method='hist'),
                'SVM': SVC(kernel="rbf", class_weight='balanced', probability=True, random_state=SEED, cache_size=2000)
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


df_raw = pd.DataFrame(results)
df_final = df_raw.groupby(['Scenario', 'Cutoff', 'Technique', 'Model']).agg({
    'MCC': 'mean', 'F1': ['mean', 'std'], 'AUC': 'mean',
}).reset_index()

df_final.columns = ['_'.join(col).strip() if col[1] else col[0] for col in df_final.columns.values]
df_final.to_csv("Fase1_Benchmark_Riduzioni.csv", index=False)