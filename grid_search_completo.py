""" 
THIS SCRIPT DOES A GRID SEARCH FOR CRC CLASSIFICATION TO FINE-TUNE THE FOLLOWING TECHNIQUES ON RF, XGB AND SVM:
- none
- SKB
- Aitchison PCA
- none (with CLR)
- RCLR
- SKB + RCLR
"""

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
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline


from utils_crc2 import * 


SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

scenarios = ['healthy']
cutoffs = [0.03, 0.05, 0.07, 0.1, 0.15, 0.2]

mcc_scorer = make_scorer(matthews_corrcoef)

pipe_none = Pipeline([
    ('scaler',StandardScaler()),
    ('classifier', None)
])

pipe_skb = Pipeline([
    ('skb', SelectKBest(score_func=mutual_info_classif)),
    ('scaler', StandardScaler()),
    ('classifier',None)
])

clr_transformer = FunctionTransformer(trasformazione_clr)

pipe_aitchison = Pipeline([
    ('clr',clr_transformer),
    ('pca',PCA(svd_solver='full',random_state=SEED)),
    ('classifier',None)
])

pipe_none_clr = Pipeline([
    ('clr',clr_transformer),
    ('scaler',StandardScaler()),
    ('classifier', None)
])

rclr_transformer = FunctionTransformer(trasformazione_rclr_nativa)

pipe_rclr = Pipeline([
    ('rclr',rclr_transformer),
    ('scaler',StandardScaler()),
    ('classifier', None)
])

pipe_rclr_skb = Pipeline([
    ('rclr',rclr_transformer),
    ('skb', SelectKBest(score_func=mutual_info_classif).set_output(transform="pandas")),
    ('scaler',StandardScaler()),
    ('classifier', None)
])

pipe_consensus = Pipeline([
    ('consensus', ConsensusFilter()),
    ('scaler', StandardScaler()),
    ('classifier', None)
])

pipelines = {
    # 'None': pipe_none,
    # 'SKB': pipe_skb,
    # 'Aitchison': pipe_aitchison,
    # 'None (CLR)': pipe_none_clr,
    # 'RCLR': pipe_rclr,
    # 'RCLR + SKB': pipe_rclr_skb,
    'Consensus' : pipe_consensus
}

models = {
    'RF': RandomForestClassifier(random_state=SEED),
    'XGB': XGBClassifier(random_state=SEED, tree_method='hist'),
    'SVM': SVC(probability=True, random_state=SEED, cache_size=2000)
}

param_grids = {
    'None': {
        'RF': {
            'classifier__n_estimators': [100, 300], 
            'classifier__max_depth': [None, 10, 20]
        },
        'XGB': {
            'classifier__n_estimators': [100, 300], 
            'classifier__learning_rate': [0.01, 0.1], 
            'classifier__max_depth': [3, 6],
            'classifier__subsample': [0.8, 1.0],         
            'classifier__colsample_bytree': [0.8, 1.0]   
        },
        'SVM': [
            {'classifier__kernel': ['rbf'], 'classifier__C': [0.1, 1, 10], 'classifier__gamma': ['scale', 'auto']},
            {'classifier__kernel': ['linear'], 'classifier__C': [0.1, 1, 10]}
        ]
    },
    'SKB': {
        'RF': {
            'skb__k': [50, 100, 200], 
            'classifier__n_estimators': [100, 300]
        },
        'XGB': {
            'skb__k': [50, 100, 200], 
            'classifier__n_estimators': [100, 300],
            'classifier__subsample': [0.8, 1.0]
        },
        'SVM': [
            {'skb__k': [50, 100, 200], 'classifier__kernel': ['rbf'], 'classifier__C': [0.1, 1, 10], 'classifier__gamma': ['scale', 'auto']},
            {'skb__k': [50, 100, 200], 'classifier__kernel': ['linear'], 'classifier__C': [0.1, 1, 10]}
        ]
    },
    'Aitchison': {
        'RF': {
            'pca__n_components': [15, 30, 0.90], 
            'classifier__n_estimators': [100, 300]
        },
        'XGB': {
            'pca__n_components': [15, 30, 0.90], 
            'classifier__n_estimators': [100, 300],
            'classifier__subsample': [0.8, 1.0]
        },
        'SVM': [
            {'pca__n_components': [15, 30, 0.90], 'classifier__kernel': ['rbf'], 'classifier__C': [0.1, 1, 10], 'classifier__gamma': ['scale']},
            {'pca__n_components': [15, 30, 0.90], 'classifier__kernel': ['linear'], 'classifier__C': [0.1, 1, 10]}
        ]
    },
    'None (CLR)': {
        'RF': {
            'classifier__n_estimators': [100, 300], 
            'classifier__max_depth': [None, 10, 20]
        },
        'XGB': {
            'classifier__n_estimators': [100, 300], 
            'classifier__learning_rate': [0.01, 0.1], 
            'classifier__max_depth': [3, 6],
            'classifier__subsample': [0.8, 1.0],         
            'classifier__colsample_bytree': [0.8, 1.0]   
        },
        'SVM': [
            {'classifier__kernel': ['rbf'], 'classifier__C': [0.1, 1, 10], 'classifier__gamma': ['scale', 'auto']},
            {'classifier__kernel': ['linear'], 'classifier__C': [0.1, 1, 10]}
        ]
    },
    'RCLR': {
        'RF': {
            'classifier__n_estimators': [100, 300], 
            'classifier__max_depth': [None, 10, 20]
        },
        'XGB': {
            'classifier__n_estimators': [100, 300], 
            'classifier__learning_rate': [0.01, 0.1], 
            'classifier__max_depth': [3, 6],
            'classifier__subsample': [0.8, 1.0],         
            'classifier__colsample_bytree': [0.8, 1.0]   
        },
        'SVM': [
            {'classifier__kernel': ['rbf'], 'classifier__C': [0.1, 1, 10], 'classifier__gamma': ['scale', 'auto']},
            {'classifier__kernel': ['linear'], 'classifier__C': [0.1, 1, 10]}
        ]
    },
    'RCLR + SKB': {
        'RF': {
            'skb__k': [50, 100, 200], 
            'classifier__n_estimators': [100, 300]
        },
        'XGB': {
            'skb__k': [50, 100, 200], 
            'classifier__n_estimators': [100, 300],
            'classifier__subsample': [0.8, 1.0]
        },
        'SVM': [
            {'skb__k': [50, 100, 200], 'classifier__kernel': ['rbf'], 'classifier__C': [0.1, 1, 10], 'classifier__gamma': ['scale', 'auto']},
            {'skb__k': [50, 100, 200], 'classifier__kernel': ['linear'], 'classifier__C': [0.1, 1, 10]}
        ]
    },
    'Consensus': {
        'RF': {
            'consensus__k': [100, 200, 300],
            'classifier__n_estimators': [100, 300], 
            'classifier__max_depth': [None, 10, 20]
        },
        'XGB': {
            'consensus__k': [100, 200, 300], 
            'classifier__n_estimators': [100, 300], 
            'classifier__learning_rate': [0.01, 0.1], 
            'classifier__max_depth': [3, 6],
            'classifier__subsample': [0.8, 1.0],         
            'classifier__colsample_bytree': [0.8, 1.0]   
        },
        'SVM': [
            {'consensus__k': [100, 200, 300], 'classifier__kernel': ['rbf'], 'classifier__C': [0.1, 1, 10], 'classifier__gamma': ['scale', 'auto']},
            {'consensus__k': [100, 200, 300], 'classifier__kernel': ['linear'], 'classifier__C': [0.1, 1, 10]}
        ]
    }
}

results = []

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

                df_results.to_csv("Fase2_Vincitori_GridSearch_Maggioranza.csv", index=False)

