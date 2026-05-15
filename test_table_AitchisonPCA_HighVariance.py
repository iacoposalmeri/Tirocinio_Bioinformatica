import warnings
# Ignora gli avvisi di UMAP (n_jobs e precomputed)
warnings.filterwarnings('ignore', category=UserWarning)
# Ignora gli avvisi di scikit-learn sui futuri aggiornamenti (copy)
warnings.filterwarnings('ignore', category=FutureWarning)
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import umap
from sklearn.cluster import HDBSCAN
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, matthews_corrcoef
from skbio.diversity import beta_diversity
from utils_crc import * 
from skbio.stats.ordination import pcoa
from skbio.diversity import beta_diversity
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from xgboost import XGBClassifier

SEED = 42

scenarios = ['control', 'healthy']

cutoffs = [0.03, 0.05, 0.07, 0.1, 0.15, 0.2]

dim_red = ['Aitchison_PCA']

results = []

for scenario in scenarios:

    x, y_binary, metadati_finali_no_desease, metadati_esclusi = caricamento_pulizia_dati("Metadati_CRC_Dataset.csv", "Abbondanze_CRC_Dataset.csv",condizione_negativa=scenario)

    X_train, X_test, y_train, y_test = train_test_split(x, y_binary, test_size=0.2, random_state=SEED, stratify=y_binary)

    for cutoff in cutoffs:

        for technique in dim_red:

            if technique == 'Aitchison_PCA':

                batteri_da_tenere = maschera_prevalenza(X_train, y_train, cutoff)

                X_train_filtrato = filtraggio(X_train, batteri_da_tenere)
                X_test_filtrato = filtraggio(X_test, batteri_da_tenere)

        
                X_train_filtrato = trasformazione_clr(X_train_filtrato)
                X_test_filtrato = trasformazione_clr(X_test_filtrato)

                X_train_scaled, X_test_scaled = standard_scaler(X_train_filtrato, X_test_filtrato)

                pca = PCA(n_components=0.90,random_state=42)

                X_train_final = pca.fit_transform(X_train_scaled) 
                X_test_final = pca.transform(X_test_scaled)

            y_train_aligned = y_train.reset_index(drop=True) 

            RF = RandomForestClassifier(random_state=SEED, class_weight='balanced')
            XGB = XGBClassifier(n_jobs=-1)
            SVM = svm.SVC(kernel="rbf", class_weight = 'balanced', probability=True, random_state=SEED, cache_size=1000)

            cv_strategy = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)

            measures = {
                'accuracy': 'accuracy',
                'f1': 'f1',
                'recall': 'recall',
                'precision': 'precision',
                'mcc': make_scorer(matthews_corrcoef),
                'auc': 'roc_auc'
            }

            rf_res = cross_validate(RF, X_train_final, y_train_aligned, cv=cv_strategy, scoring=measures)
            xgb_res = cross_validate(XGB, X_train_final, y_train_aligned, cv=cv_strategy, scoring=measures)
            svm_res = cross_validate(SVM, X_train_final, y_train_aligned, cv=cv_strategy, scoring=measures)

            modelli_eval = {
                'RandomForest': rf_res,
                'XGBoost': xgb_res,
                'SVM': svm_res
            }
            
            for nome_modello, res in modelli_eval.items():
                results.append({
                    'Scenario': scenario,
                    'Cutoff': cutoff,
                    'Technique': technique,
                    'Model': nome_modello,
                    'F1_Mean': res['test_f1'].mean(),
                    'F1_Std': res['test_f1'].std(),
                    'MCC_Mean': res['test_mcc'].mean(),
                    'Accuracy_Mean': res['test_accuracy'].mean(),
                    'AUC_Mean': res['test_auc'].mean(),
                    'Recall_Mean': res['test_recall'].mean(),
                    'Precision_Mean': res['test_precision'].mean()
                })



df_final = pd.DataFrame(results)
df_final.to_csv("Risultati_Esplorazione_AITCHISON_PCA_HIGH_VARIANCE.csv", index=False)