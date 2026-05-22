import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*random_state does not influence oneDAL.*')
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
from sklearn.feature_selection import SelectKBest, mutual_info_classif

from sklearnex import patch_sklearn
patch_sklearn()

SEED = 42

scenarios = ['control', 'healthy']

cutoffs = [0.03, 0.05, 0.07, 0.1, 0.15, 0.2]

dim_red = ['RPCA']

""" x, y_binary, metadati_finali_no_desease, metadati_esclusi = caricamento_pulizia_dati("Metadati_CRC_Dataset.csv", "Abbondanze_CRC_Dataset.csv",condizione_negativa='control')
X_train, X_test, y_train, y_test = train_test_split(x, y_binary, test_size=0.2, random_state=SEED, stratify=y_binary)

batteri_da_tenere = maschera_prevalenza(X_train, y_train, 0.03)

X_train_filtrato = filtraggio(X_train, batteri_da_tenere)
X_test_filtrato = filtraggio(X_test, batteri_da_tenere)

X_train_biom = table.Table(X_train_filtrato.values.T, observation_ids=X_train_filtrato.columns, sample_ids=X_train_filtrato.index)

ordination, distance_matrix = rpca(X_train_biom, n_components=5)

# SCATTERPLOT

fig1,ax1 = plt.subplots(figsize=(18,16))

sns.scatterplot(
    x=ordination.samples['PC1'],
    y=ordination.samples['PC2'],
    hue = ordination.samples.index.map(y_train.map({0: "Control", 1: "CRC"})),
    palette = {'Control' : "#F46E27", 'CRC' : "#1C5CDC"},
    ax = ax1,
    alpha = 0.7,
    s = 50
)

var_pc1 = ordination.proportion_explained.iloc[0] * 100
var_pc2 = ordination.proportion_explained.iloc[1] * 100

ax1.set_title(f"Cutoff {0.03*100:.0f}%", fontweight='bold', fontsize=12)
ax1.set_xlabel(f"PC1 ({var_pc1:.1f}%)")
ax1.set_ylabel(f"PC2 ({var_pc2:.1f}%)")

fig1.savefig("Grid_RPCA.png", dpi=300, bbox_inches='tight')

# SCREEPLOT

fig_elb, ax_elb = plt.subplots(figsize=(18,10))

variance_elb = ordination.proportion_explained * 100
cumulative_variance_elb = np.cumsum(variance_elb)

ax_elb.bar(range(1,len(variance_elb)+1), variance_elb, alpha=0.6, align='center', label='Varianza Singola', color='#3498db')
ax_elb.step(range(1,len(cumulative_variance_elb)+1),cumulative_variance_elb, where='mid', label='Varianza Cumulata', color='#e74c3c', marker='o', markersize=4)

ax_elb.set_ylabel('Varianza Spiegata (%)', fontsize=12)
ax_elb.set_xlabel('Indice della Componente Principale (PC)', fontsize=12)
ax_elb.set_title('Scree Plot: Ricerca del Gomito (Elbow Method)', fontsize=14, fontweight='bold')
ax_elb.set_xticks(np.arange(1, 6, step=1)) 
ax_elb.legend(loc='center right')
ax_elb.grid(True, linestyle='--', alpha=0.5)

fig_elb.savefig("Explained_Variance_RPCA.png", dpi=300, bbox_inches='tight') """

results = []

n_components = [15]

for scenario in scenarios:

    x, y_binary, metadati_finali_no_desease, metadati_esclusi = caricamento_pulizia_dati("Metadati_CRC_Dataset.csv", "Abbondanze_CRC_Dataset.csv",condizione_negativa=scenario)

    X_train, X_test, y_train, y_test = train_test_split(x, y_binary, test_size=0.2, random_state=SEED, stratify=y_binary)

    for cutoff in cutoffs:

        for technique in dim_red:

            for n_comp in n_components:

                if technique == 'RPCA':

                    batteri_da_tenere = maschera_prevalenza(X_train, y_train, cutoff)

                    X_train_filtrato = filtraggio(X_train, batteri_da_tenere)
                    X_test_filtrato = filtraggio(X_test, batteri_da_tenere)

                    SKB = SelectKBest(score_func=mutual_info_classif, k=100)

                    X_train_skb = SKB.fit_transform(X_train_filtrato, y_train)

                    X_test_skb = SKB.transform(X_test_filtrato)

                    skb_columns = SKB.get_feature_names_out(X_train_filtrato.columns)

                    X_train_filtrato = pd.DataFrame(X_train_skb, index=X_train_filtrato.index, columns=skb_columns)
                    X_test_filtrato = pd.DataFrame(X_test_skb, index=X_test_filtrato.index, columns=skb_columns)

                    X_train_biom = table.Table(X_train_filtrato.values.T, observation_ids=X_train_filtrato.columns, sample_ids=X_train_filtrato.index)

                    ordination, distance_matrix = rpca(X_train_biom, n_components=n_comp)

                    scaler = StandardScaler()

                    X_train_final = scaler.fit_transform(ordination.samples.values)
    

                y_train_aligned = y_train.reset_index(drop=True) 

                RF = RandomForestClassifier(n_jobs=-1, random_state=SEED, class_weight='balanced')
                XGB = XGBClassifier(random_state=SEED, n_jobs=-1, tree_method='hist')
                SVM = svm.SVC(kernel="rbf", class_weight = 'balanced', probability=True, random_state=SEED, cache_size=2000)

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
                        'n_comp': n_comp,
                        'F1_Mean': res['test_f1'].mean(),
                        'F1_Std': res['test_f1'].std(),
                        'MCC_Mean': res['test_mcc'].mean(),
                        'Accuracy_Mean': res['test_accuracy'].mean(),
                        'AUC_Mean': res['test_auc'].mean(),
                        'Recall_Mean': res['test_recall'].mean(),
                        'Precision_Mean': res['test_precision'].mean()
                    })



df_final = pd.DataFrame(results)
df_final.to_csv("Risultati_Esplorazione_RPCA.csv", index=False)