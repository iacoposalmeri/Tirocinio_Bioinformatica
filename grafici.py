import warnings
warnings.filterwarnings('ignore')

import os
import random
import ast

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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

df_grid = pd.read_csv('Gridsearch_finale.csv')

df_grid = df_grid.sort_values(by='Test_MCC', ascending=False).reset_index(drop=True)

print("Dati riparati e ordinati con successo!")
print(df_grid.info())
print("\nI Top 5 Veri:")
print(df_grid[['Technique', 'Model', 'Cutoff', 'Test_MCC', 'Test_F1']].head(5))


SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

scenario = 'healthy'

x, y_binary, _, _ = caricamento_pulizia_dati("Metadati_CRC_Dataset.csv", "Abbondanze_CRC_Dataset.csv", condizione_negativa=scenario)
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(x, y_binary, test_size=0.2, random_state=SEED, stratify=y_binary)


print("\nEstraendo il numero di feature per il grafico...")

mappa_feature_post_cutoff = {
    0.03: 392,
    0.05: 317,
    0.07: 276,
    0.10: 240,
    0.15: 208,
    0.20: 174
}

tqdm.pandas(desc="Simulazione Consensus in corso")

def calcola_feature_reali(row):
    tecnica = str(row['Technique'])
    
    try:
        params = ast.literal_eval(row['Best_Params'])
    except:
        params = {}

    if 'Consensus' not in tecnica:
        if 'rfe__n_features_to_select' in params: return params['rfe__n_features_to_select']
        if 'skb__k' in params: return params['skb__k']
        if 'elasticnet__max_features' in params: return params['elasticnet__max_features']
        # Fallback se non c'è Feature Selection (es. Baseline)
        c = float(row['Cutoff'])
        return mappa_feature_post_cutoff.get(c, None)

    cutoff = float(row['Cutoff'])
    k_richiesto = params.get('consensus__k', 50)
    threshold = params.get('consensus__threshold', 2)
    
    prevalenza = (X_train_full > 0).mean()
    batteri_validi = prevalenza[prevalenza >= cutoff].index
    X_train_filt = X_train_full[batteri_validi]
    
    if 'RCLR' in tecnica:
        X_train_trasf = trasformazione_rclr_nativa(X_train_filt)
    else:
        X_train_trasf = trasformazione_clr(X_train_filt)
        
    k_effettivo = min(X_train_trasf.shape[1], k_richiesto)
    
    skb = SelectKBest(score_func=mutual_info_classif, k=k_effettivo)
    skb.fit(X_train_trasf, y_train_full)
    voti_skb = list(skb.get_support(indices=True))
    
    stimatore_rfe = RandomForestClassifier(n_estimators=50, random_state=SEED, n_jobs=-1)
    rfe = RFE(estimator=stimatore_rfe, n_features_to_select=k_effettivo, step=0.1)
    rfe.fit(X_train_trasf, y_train_full)
    voti_rfe = list(rfe.get_support(indices=True))
    
    modello_en = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, random_state=SEED, max_iter=1000)
    selettore_en = SelectFromModel(modello_en, max_features=k_effettivo, prefit=False)
    selettore_en.fit(X_train_trasf, y_train_full)
    voti_elan = list(selettore_en.get_support(indices=True))
    
    tutti_i_voti = voti_skb + voti_rfe + voti_elan
    conteggio = Counter(tutti_i_voti)
    indici_vincitori = [indice for indice, voti in conteggio.items() if voti >= threshold]
    
    if len(indici_vincitori) == 0:
        return len(voti_skb)
    else:
        return len(indici_vincitori)

def estrai_k(row):
    try:
        params = ast.literal_eval(row['Best_Params'])
        if 'rfe__n_features_to_select' in params: return params['rfe__n_features_to_select']
        if 'skb__k' in params: return params['skb__k']
        if 'elasticnet__max_features' in params: return params['elasticnet__max_features']
        if 'consensus__k' in params: return params['consensus__k']
    except:
        pass

    cutoff_della_riga = row['Cutoff']
    if cutoff_della_riga in mappa_feature_post_cutoff:
        return mappa_feature_post_cutoff[cutoff_della_riga]
        
    return None

df_grid['N_Features'] = df_grid.progress_apply(calcola_feature_reali, axis=1)

print("Creazione del Trade-Off Plot")

sns.set_theme(style='whitegrid',context='paper', font_scale=1.2)
plt.figure(figsize=(10,6))

sns.scatterplot(
    data=df_grid,
    x='N_Features',
    y='Test_MCC',
    hue='Technique',
    style='Model',
    s=120,
    alpha=0.8,
    palette='deep'
)

plt.title("Trade-off: Dimensionalità vs Potere Predittivo", fontsize=15, fontweight='bold', pad=15)
plt.xlabel("Numero di Feature Selezionate", fontsize=12, fontweight='bold')
plt.ylabel("Matthews Correlation Coefficient (Test MCC)", fontsize=12, fontweight='bold')

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Legenda", title_fontsize='11', fontsize='10')

plt.tight_layout()
nome_grafico = "Grafico1_TradeOff.png"
plt.savefig(nome_grafico, dpi=300, bbox_inches='tight')
plt.close()

print(f"Grafico salvato come: {nome_grafico}")

print("Creazione del BoxPlot della Stabilità")

sns.set_theme(style='whitegrid',context='paper', font_scale=1.2)
plt.figure(figsize=(10,6))

ordine_tecniche = df_grid.groupby('Technique')['CV_MCC_Score'].mean().sort_values(ascending=False).index

sns.boxplot(
    data=df_grid,
    x='Technique',
    y='CV_MCC_Score',
    order=ordine_tecniche,
    palette='pastel',
    hue='Technique',
    showfliers=False
)

sns.stripplot(
    data=df_grid,
    x='Technique',
    y='CV_MCC_Score',
    order=ordine_tecniche,
    hue='Model',
    palette='dark',
    dodge=True,
    alpha=0.6,
    jitter=True
)

plt.xticks(fontsize=9)

plt.title("Stabilità Algoritmica in Cross-Validation (CV MCC)", fontsize=15, fontweight='bold', pad=15)
plt.xlabel("Tecnica di Estrazione/Selezione Spaziale", fontsize=12, fontweight='bold')
plt.ylabel("MCC Medio (10-Fold Cross Validation)", fontsize=12, fontweight='bold')

plt.legend(title='Algoritmo di Classificazione', loc='best')

plt.tight_layout()
nome_grafico2 = "Grafico2_BoxPlot.png"
plt.savefig(nome_grafico2, dpi=300, bbox_inches='tight')
plt.close()

print(f"Grafico salvato come: {nome_grafico2}")

print("Creazione del Grafico sull' Andamento del Cutoff")

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.figure(figsize=(10, 6))

sns.lineplot(
    data=df_grid,
    x='Cutoff',
    y='Test_MCC',
    hue='Model',
    style='Model',
    markers=True,
    dashes=False,
    linewidth=2.5,
    markersize=9,
    palette='deep'
)

plt.title("Impatto del Livello di Filtraggio (Cutoff) sulle Performance", fontsize=15, fontweight='bold', pad=15)
plt.xlabel("Soglia di Cutoff (Prevalenza minima richiesta)", fontsize=12, fontweight='bold')
plt.ylabel("MCC Medio sul Test Set", fontsize=12, fontweight='bold')

valori_cutoff = sorted(df_grid['Cutoff'].unique())
plt.xticks(valori_cutoff, [str(c) for c in valori_cutoff])

plt.legend(title="Algoritmo", loc='lower right')

plt.tight_layout()
nome_grafico3 = "Grafico3_Andamento_Cutoff.png"
plt.savefig(nome_grafico3, dpi=300, bbox_inches='tight')
plt.close()

print(f"Grafico salvato come: {nome_grafico3}")

print("Creazione dell'Heatmap")

sns.set_theme(style="white", context="paper", font_scale=1.2)
plt.figure(figsize=(10, 4))

matrice_mcc = df_grid.pivot_table(
    index='Model', 
    columns='Technique', 
    values='Test_MCC', 
    aggfunc='max'
)

sns.heatmap(
    data=matrice_mcc,
    annot=True,
    fmt=".3f",
    linewidth=.5,
    cmap='YlOrRd',
    cbar_kws={'label': 'MCC Massimo Raggiunto'}
)

plt.title("Mappa delle Sinergie: Algoritmo vs Feature Selection", fontsize=15, fontweight='bold', pad=15)
plt.xlabel("Tecnica di Estrazione/Selezione", fontsize=12, fontweight='bold')
plt.ylabel("Modello Predittivo", fontsize=12, fontweight='bold')

plt.xticks(rotation=45, ha='right')

plt.tight_layout()
nome_grafico4 = "Grafico4_Heatmap.png"
plt.savefig(nome_grafico4, dpi=300, bbox_inches='tight')
plt.close()

print(f"Grafico salvato come: {nome_grafico4}")

print("Generando il Grafico 5: Il Podio dei Finalisti...")


df_top5 = df_grid.sort_values(by='Test_MCC', ascending=False).head(5).copy()

df_top5['Nome_Modello'] = df_top5.apply(
    lambda r: f"{r['Technique']} ({r['Model']})\nCutoff: {r['Cutoff']} | Feat: {r['N_Features']}", axis=1
)


df_melted = df_top5.melt(
    id_vars=['Nome_Modello'], 
    value_vars=['Test_MCC', 'Test_F1', 'Test_AUC'],
    var_name='Metrica', 
    value_name='Punteggio'
)

df_melted['Metrica'] = df_melted['Metrica'].replace({
    'Test_MCC': 'MCC (Robustezza)', 
    'Test_F1': 'F1-Score (Precisione)', 
    'Test_AUC': 'AUC (Separazione)'
})

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.figure(figsize=(12, 7))

# Creiamo il Barplot
ax = sns.barplot(
    data=df_melted, 
    x='Nome_Modello', 
    y='Punteggio', 
    hue='Metrica', 
    palette=['#2ca02c', '#1f77b4', '#ff7f0e', "#ff0eef", "#ffe30e"],
    edgecolor='black',
    linewidth=1
)

for container in ax.containers:
    ax.bar_label(container, fmt='%.3f', padding=3, fontsize=11, fontweight='bold')

plt.title("Podio dei Biomarcatori: Performance Finali sul Test Set", fontsize=16, fontweight='bold', pad=20)
plt.xlabel("Specifiche del Pannello Diagnostico", fontsize=13, fontweight='bold')
plt.ylabel("Punteggio (0.0 - 1.0)", fontsize=13, fontweight='bold')

plt.ylim(0, df_melted['Punteggio'].max() + 0.15) 

plt.legend(title="Metrica Clinica", loc='best', framealpha=0.8)

plt.tight_layout()
nome_grafico5 = "Grafico5_Podio_Finale.png"
plt.savefig(nome_grafico5, dpi=300, bbox_inches='tight')
plt.close()

print(f"Grafico salvato come: {nome_grafico5}")