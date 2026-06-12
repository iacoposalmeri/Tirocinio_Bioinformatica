import ast
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

scenario = 'healthy'

df = pd.read_csv("Gridsearch_finale.csv")

prime_5_righe = df.sort_values('Test_MCC', ascending=False).head(5).reset_index(drop=True)

x, y_binary, _, _ = caricamento_pulizia_dati("Metadati_CRC_Dataset.csv", "Abbondanze_CRC_Dataset.csv",condizione_negativa=scenario)

X_train, _, y_train, _ = train_test_split(x, y_binary, test_size=0.2, random_state=SEED, stratify=y_binary)


dizionario_biomarcatori = {}

for index, row in prime_5_righe.iterrows():
    tecnica = row['Technique']
    modello = row['Model']
    cutoff = float(row['Cutoff'])
    params = ast.literal_eval(row['Best_Params'])

    nome_esperimento = f"{tecnica}_{modello}_Cutoff{cutoff}_Rank{index+1}"
    print(f"\n{'='*60}\nRank {index+1}: {nome_esperimento}")

    maschera = maschera_prevalenza(X_train,y_train,cutoff=cutoff)
    X_train_filtrato = filtraggio(X_train, maschera)
    print(f"Batteri post-cutoff ({cutoff}): {len(maschera)}")

    if 'RCLR' in tecnica:
        X_train_trasf = trasformazione_rclr_nativa(X_train_filtrato)
    else:
        X_train_trasf = trasformazione_clr(X_train_filtrato)

    nomi_batteri = X_train_trasf.columns

    if 'Consensus' in tecnica:
        k = params.get('consensus__k', 50)
        threshold = params.get('consensus__threshold', 2)
        print(f"Modalità Consensus (k={k}, threshold={threshold})")
        
        k_effettivo = min(X_train_trasf.shape[1], k)
        
        # 1. SKB
        skb = SelectKBest(score_func=mutual_info_classif, k=k_effettivo)
        skb.fit(X_train_trasf, y_train)
        voti_skb = set(nomi_batteri[skb.get_support(indices=True)])
        
        # 2. RFE
        stimatore_rfe = RandomForestClassifier(n_estimators=50, random_state=SEED, n_jobs=-1)
        rfe = RFE(estimator=stimatore_rfe, n_features_to_select=k_effettivo, step=0.1)
        rfe.fit(X_train_trasf, y_train)
        voti_rfe = set(nomi_batteri[rfe.get_support(indices=True)])
        
        # 3. ElasticNet
        modello_en = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, random_state=SEED, max_iter=1000)
        selettore_en = SelectFromModel(modello_en, max_features=k_effettivo, prefit=False)
        selettore_en.fit(X_train_trasf, y_train)
        voti_elan = set(nomi_batteri[selettore_en.get_support(indices=True)])
        

        unione = voti_skb.union(voti_rfe).union(voti_elan)
        intersezione = voti_skb.intersection(voti_rfe).intersection(voti_elan)
        tutti_i_voti_list = list(voti_skb) + list(voti_rfe) + list(voti_elan)
        
        maggioranza = set([b for b in unione if tutti_i_voti_list.count(b) >= threshold])
        
        if len(maggioranza) == 0:
            print("  [!] Nessun batterio ha superato la soglia. Fallback su SKB.")
            feature_finali = list(voti_skb)
        else:
            feature_finali = list(maggioranza)
            
        # print(f"  -> Trovati da SKB: {len(voti_skb)}")
        # print(f"  -> Trovati da RFE: {len(voti_rfe)}")
        # print(f"  -> Trovati da E-Net: {len(voti_elan)}")
        # print(f"  -> INTERSEZIONE: {len(intersezione)}")
        # print(f"  -> UNIONE: {len(unione)}")
        # print(f"  -> MAGGIORANZA (>= {threshold} voti): {len(maggioranza)}")
        
        dizionario_biomarcatori[nome_esperimento] = feature_finali
        
        df_cons = pd.DataFrame({
            'Batterio': list(unione),
            'Votato_da_SKB': [b in voti_skb for b in unione],
            'Votato_da_RFE': [b in voti_rfe for b in unione],
            'Votato_da_ElasticNet': [b in voti_elan for b in unione],
            'Totale_Voti': [tutti_i_voti_list.count(b) for b in unione],
            'Scelto_in_Finale': [b in feature_finali for b in unione]
        })
        nome_file_cons = f"Consensus_Autopsia_Rank{index+1}.csv"
        df_cons.to_csv(nome_file_cons, index=False)
        print(f"  [!] Dettaglio voti salvato in: {nome_file_cons}")

    else:
        if 'RFE' in tecnica:
            k = params.get('rfe__n_features_to_select', 50)
            k_effettivo = min(X_train_trasf.shape[1], k)
            base_model = RandomForestClassifier(random_state=SEED) if modello == 'RF' else XGBClassifier(random_state=SEED, eval_metric='logloss')
            selector = RFE(estimator=base_model, n_features_to_select=k_effettivo)
        elif 'SKB' in tecnica:
            k = params.get('skb__k', 50)
            k_effettivo = min(X_train_trasf.shape[1], k)
            selector = SelectKBest(score_func=mutual_info_classif, k=k_effettivo)
        elif 'ElasticNet' in tecnica:
            k = params.get('elasticnet__max_features', 50)
            k_effettivo = min(X_train_trasf.shape[1], k)
            selector = SelectFromModel(LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, random_state=SEED), max_features=k_effettivo)
            
        print(f"Estrazione Standard ({tecnica}) - Cerca k={k_effettivo}")
        selector.fit(X_train_trasf, y_train)
        feature_scelte = list(nomi_batteri[selector.get_support(indices=True)])
        dizionario_biomarcatori[nome_esperimento] = feature_scelte
        print(f"  -> Estrazione completata: {len(feature_scelte)} batteri trovati.")


max_len = max([len(v) for v in dizionario_biomarcatori.values()])

for k in dizionario_biomarcatori:
    dizionario_biomarcatori[k].extend([np.nan] * (max_len - len(dizionario_biomarcatori[k])))

df_biomarcatori = pd.DataFrame(dizionario_biomarcatori)
df_biomarcatori.to_csv("Top5_Biomarcatori_Estratti.csv", index=False)

print(f"\n{'='*60}")
print("Tutti i biomarcatori sono stati estratti e salvati in 'Top5_Biomarcatori_Estratti.csv'")