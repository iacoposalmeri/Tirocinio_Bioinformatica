

import warnings
warnings.filterwarnings('ignore')

import os
import random

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import matthews_corrcoef, f1_score, roc_auc_score, make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE, SelectFromModel
from sklearn.pipeline import Pipeline


from utils_crc2 import * 


SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

scenarios = ['healthy']
cutoffs = [0.03, 0.05, 0.07, 0.1, 0.15, 0.2]

mcc_scorer = make_scorer(matthews_corrcoef)

clr_transformer = FunctionTransformer(trasformazione_clr)
rclr_transformer = FunctionTransformer(trasformazione_rclr_nativa)
pipe_clr_rfe = Pipeline([
    ('clr',clr_transformer),
    ('rfe', RFE(estimator=RandomForestClassifier(n_estimators=50, random_state=SEED, n_jobs=-1), step=0.1)),
    ('scaler',StandardScaler()),
    ('classifier', None)
])

pipe_rclr_rfe = Pipeline([
    ('rclr',rclr_transformer),
    ('rfe', RFE(estimator=RandomForestClassifier(n_estimators=50, random_state=SEED), step=0.1)),
    ('scaler',StandardScaler()),
    ('classifier', None)
])

pipelines = {
    'CLR_RFE': pipe_clr_rfe,
    'RCLR_RFE': pipe_rclr_rfe
}

models = {
    'RF': RandomForestClassifier(random_state=SEED)}


param_grids = {
    'CLR_RFE': {
        'RF': {
            'rfe__n_features_to_select': [20, 50, 100, 200], 
            'classifier__n_estimators': [100, 300],
            'classifier__max_depth': [None, 10],
            'classifier__max_features': ['sqrt', 'log2'],
            'classifier__min_samples_leaf': [1, 3]
        }
    },
    'RCLR_RFE': {
        'RF': {
            'rfe__n_features_to_select': [20, 50, 100, 200], 
            'classifier__n_estimators': [100, 300],
            'classifier__max_depth': [None, 10],
            'classifier__max_features': ['sqrt', 'log2'],
            'classifier__min_samples_leaf': [1, 3]
        }
    }
}   

results = []

if __name__ == '__main__':
    
    for scenario in tqdm(scenarios, desc="Scenari"):
        
        x, y_binary, _, _ = caricamento_pulizia_dati("Metadati_CRC_Dataset.csv", "Abbondanze_CRC_Dataset.csv", condizione_negativa=scenario)
        X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(x, y_binary, test_size=0.2, random_state=SEED, stratify=y_binary)
        
        for cutoff in tqdm(cutoffs, desc=f"Cutoffs per {scenario}", leave=False):
            
            bacteria_to_keep = maschera_prevalenza(X_train_full, y_train_full, cutoff=cutoff)
            X_train_filt = filtraggio(X_train_full, bacteria_to_keep)
            X_test_filt = filtraggio(X_test_full, bacteria_to_keep)
            
            mask_validi = (X_train_filt != 0).any(axis=1)
            X_train_filt = X_train_filt[mask_validi]
            y_train = y_train_full[mask_validi]
            
            X_test_filt = X_test_filt[X_train_filt.columns]

            cv_inner = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)

            for pipe_name, pipeline in pipelines.items():
                for model_name, model in models.items():
                    
                    pipeline.set_params(classifier=model)
                    grid = param_grids[pipe_name][model_name]
                    
                    search = GridSearchCV(
                        pipeline, 
                        grid, 
                        cv=cv_inner, 
                        scoring=mcc_scorer, 
                        refit=True,
                        n_jobs=-1,
                        verbose=1
                    )
                    
                    search.fit(X_train_filt, y_train)
                    
                    best_model = search.best_estimator_
                    y_pred = best_model.predict(X_test_filt)
                    y_proba = best_model.predict_proba(X_test_filt)[:, 1]
                    
                    results.append({
                        'Scenario': scenario,
                        'Cutoff': cutoff,
                        'Technique': pipe_name,
                        'Model': model_name,
                        'Best_Params': str(search.best_params_),
                        'CV_MCC_Score': search.best_score_,
                        'Test_MCC': matthews_corrcoef(y_test_full, y_pred),
                        'Test_F1': f1_score(y_test_full, y_pred),
                        'Test_AUC': roc_auc_score(y_test_full, y_proba)
                    })

df_results = pd.DataFrame(results)

df_results = df_results.sort_values(by=['Scenario', 'Test_MCC'], ascending=[True, False])

colonne_ordinate = ['Scenario', 'Cutoff', 'Technique', 'Model', 'Test_MCC', 'Test_F1', 'Test_AUC', 'CV_MCC_Score', 'Best_Params']
df_results = df_results[colonne_ordinate]

df_results.to_csv("GridSearch_rfe_RF.csv", index=False)