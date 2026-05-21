import warnings
from datetime import datetime # <-- Aggiunto per stampare l'orario
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, matthews_corrcoef
from utils_crc2 import * 
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from xgboost import XGBClassifier
from gemelli.rpca import rpca
from biom import table
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from sklearnex import patch_sklearn
patch_sklearn()

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*random_state does not influence oneDAL.*')

SEED = 42
scenarios = ['healthy']
cutoffs = [0.05]
dim_red = ['RPCA']
n_components = [50]

results = []

print(f"[{datetime.now().strftime('%H:%M:%S')}] Inizio elaborazione pipeline totale...")

for scenario in scenarios:
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] >>> SCENARIO: {scenario.upper()} <<<")
    
    x, y_binary, metadati_finali_no_desease, metadati_esclusi = caricamento_pulizia_dati(
        "Metadati_CRC_Dataset.csv", "Abbondanze_CRC_Dataset.csv", condizione_negativa=scenario
    )

    X_train, X_test, y_train, y_test = train_test_split(x, y_binary, test_size=0.2, random_state=SEED, stratify=y_binary)
    y_train_aligned = y_train.reset_index(drop=True) 

    for cutoff in cutoffs:
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   -> Calcolo filtraggio per Cutoff: {cutoff}")
        
        batteri_da_tenere = maschera_prevalenza(X_train, y_train, cutoff)
        X_train_filtrato = filtraggio(X_train, batteri_da_tenere)
        X_train_biom = table.Table(X_train_filtrato.values.T, observation_ids=X_train_filtrato.columns, sample_ids=X_train_filtrato.index)

        for technique in dim_red:
            for n_comp in n_components:
                print(f"[{datetime.now().strftime('%H:%M:%S')}]      * Esecuzione {technique} con {n_comp} componenti...")

                if technique == 'RPCA':
                    ordination, distance_matrix = rpca(X_train_biom, n_components=n_comp)
                    scaler = StandardScaler()
                    X_train_final = scaler.fit_transform(ordination.samples.values)
                else:
                    continue 

                # Inizializzazione modelli (n_jobs=-1 solo qui)
                RF = RandomForestClassifier(n_jobs=-1, random_state=SEED, class_weight='balanced')
                XGB = XGBClassifier(random_state=SEED, n_jobs=-1, tree_method='hist')
                SVM = svm.SVC(kernel="rbf", class_weight='balanced', probability=True, random_state=SEED, cache_size=2000)

                cv_strategy = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)

                measures = {
                    'accuracy': 'accuracy',
                    'f1': 'f1',
                    'recall': 'recall',
                    'precision': 'precision',
                    'mcc': make_scorer(matthews_corrcoef),
                    'auc': 'roc_auc'
                }

                print(f"[{datetime.now().strftime('%H:%M:%S')}]        - Cross-validation per Random Forest...")
                rf_res = cross_validate(RF, X_train_final, y_train_aligned, cv=cv_strategy, scoring=measures)
                
                print(f"[{datetime.now().strftime('%H:%M:%S')}]        - Cross-validation per XGBoost...")
                xgb_res = cross_validate(XGB, X_train_final, y_train_aligned, cv=cv_strategy, scoring=measures)
                
                print(f"[{datetime.now().strftime('%H:%M:%S')}]        - Cross-validation per SVM...")
                svm_res = cross_validate(SVM, X_train_final, y_train_aligned, cv=cv_strategy, scoring=measures)

                modelli_eval = {
                    'RandomForest': rf_res,
                    'XGBoost': xgb_res,
                    'SVM': svm_res
                }
                
                # Salvataggio risultati in memoria
                for nome_modello, res in modelli_eval.items():
                    results.append({
                        'Scenario': scenario,
                        'Cutoff': cutoff,
                        'Technique': technique,
                        'Model': nome_modello,
                        'n_comp': n_comp,
                        'F1_Mean': res['test_f1'].mean(),
                        'F1_Std': res['test_f1'].std(),
                        'MCC_Mean': res['test_mcc'].mean(),
                        'Accuracy_Mean': res['test_accuracy'].mean(),
                        'AUC_Mean': res['test_auc'].mean(),
                        'Recall_Mean': res['test_recall'].mean(),
                        'Precision_Mean': res['test_precision'].mean()
                    })

print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Elaborazione completata! Salvataggio risultati su CSV...")
df_final = pd.DataFrame(results)
df_final.to_csv("Risultati_Esplorazione_RPCA.csv", index=False)
print(f"[{datetime.now().strftime('%H:%M:%S')}] File salvato con successo.")