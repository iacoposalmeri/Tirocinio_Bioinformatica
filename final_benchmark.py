""" 
THIS SCRIPT IS A BENCHMARK FOR CRC CLASSIFICATION WITH THE ADDITIONAL FOLLOWING TECHNIQUES:
- none
- SKB
- PCA
- Aitchison PCA
- PCoA with Bray-Curtis
- RPCA
- SKB + RPCA 

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
scenarios = ['control', 'healthy']
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

            # none
            scaler_base = StandardScaler()
            reps_train['none'] = scaler_base.fit_transform(X_tr_filt)
            reps_val['none'] = scaler_base.transform(X_val_filt)

            # SKB
            skb = SelectKBest(score_func=mutual_info_classif, k=min(K_BEST_FEATURES, X_tr_filt.shape[1]))
            X_tr_skb_raw = skb.fit_transform(X_tr_filt, y_tr)
            X_val_skb_raw = skb.transform(X_val_filt)
            
            scaler_skb = StandardScaler()
            reps_train['SKB'] = scaler_skb.fit_transform(X_tr_skb_raw)
            reps_val['SKB'] = scaler_skb.transform(X_val_skb_raw)

            # PCA
            pca = PCA(n_components=N_COMPONENTS, svd_solver='full', random_state=SEED)
            scaler_pre = StandardScaler()
            scaler_post = StandardScaler()

            X_tr_scaled = scaler_pre.fit_transform(X_tr_filt)
            X_tr_pca = pca.fit_transform(X_tr_scaled)
            reps_train['PCA'] = scaler_post.fit_transform(X_tr_pca)

            X_val_scaled = scaler_pre.transform(X_val_filt)
            X_val_pca = pca.transform(X_val_scaled)
            reps_val['PCA'] = scaler_post.transform(X_val_pca)

            # AITCHISON PCA
            pca_aitchison = PCA(n_components=N_COMPONENTS, svd_solver='full', random_state=SEED)
            scaler_pca_aitchison = StandardScaler()

            X_tr_clr = trasformazione_clr(X_tr_filt)
            X_tr_pca_aitchison = pca_aitchison.fit_transform(X_tr_clr)
            reps_train['PCA_Aitchison'] = scaler_pca_aitchison.fit_transform(X_tr_pca_aitchison)

            X_val_clr = trasformazione_clr(X_val_filt)
            X_val_pca_aitchison = pca_aitchison.transform(X_val_clr)
            reps_val['PCA_Aitchison'] = scaler_pca_aitchison.transform(X_val_pca_aitchison)

            # PCoA (Bray-Curtis)
            reps_train['Bray_Curtis'] = pairwise_distances(X_tr_filt, metric='braycurtis')
            reps_val['Bray_Curtis'] = pairwise_distances(X_val_filt, X_tr_filt, metric='braycurtis')

            # RPCA
            X_tr_rpca_safe = X_tr_filt.copy()
            mask_zeri_rpca = (X_tr_rpca_safe.sum(axis=1) == 0)
            if mask_zeri_rpca.any():
                X_tr_rpca_safe.loc[mask_zeri_rpca] = 1e-9

            X_tr_biom = table.Table(X_tr_rpca_safe.values.T, observation_ids=X_tr_rpca_safe.columns, sample_ids=X_tr_rpca_safe.index)

            ordination, _ = rpca(X_tr_biom, n_components=min(N_COMPONENTS, X_tr_filt.shape[1]-1))

            val_rpca_projected = X_val_clr @ ordination.features.values

            scaler_rpca = StandardScaler()
            reps_train['RPCA'] = scaler_rpca.fit_transform(ordination.samples.values)
            reps_val['RPCA'] = scaler_rpca.transform(val_rpca_projected)


            # SKB + RPCA
            skb_columns = skb.get_feature_names_out(X_tr_filt.columns)

            X_tr_skb_df = pd.DataFrame(skb.transform(X_tr_filt), index=X_tr_filt.index, columns=skb_columns)
            X_val_skb_df = pd.DataFrame(skb.transform(X_val_filt), index=X_val_filt.index, columns=skb_columns)

            X_tr_skb_safe = X_tr_skb_df.copy()
            mask_zeri_skb = (X_tr_skb_safe.sum(axis=1) == 0)
            if mask_zeri_skb.any():
                X_tr_skb_safe.loc[mask_zeri_skb] = 1e-9

            X_tr_biom_skb = table.Table(X_tr_skb_safe.values.T, observation_ids=X_tr_skb_safe.columns, sample_ids=X_tr_skb_safe.index)

            rpca_n_comp = min(N_COMPONENTS, len(skb_columns)-1)
            ordination_skb, _ = rpca(X_tr_biom_skb, n_components=rpca_n_comp)

            X_val_skb_clr = trasformazione_clr(X_val_skb_df)
            val_rpca_projected_skb = X_val_skb_clr @ ordination_skb.features.values

            scaler_rpca_skb = StandardScaler()
            reps_train['SKB_RPCA'] = scaler_rpca_skb.fit_transform(ordination_skb.samples.values)
            reps_val['SKB_RPCA'] = scaler_rpca_skb.transform(val_rpca_projected_skb)
            
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
