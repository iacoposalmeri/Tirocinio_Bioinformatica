import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from skbio.stats.composition import clr, multiplicative_replacement as multi_replace

import numpy as np

""" def multi_replace(matrice, delta=1e-9):
    matrice_arr = np.array(matrice, dtype=float)
    matrice_arr[matrice_arr == 0] = delta
    return matrice_arr
 """
from sklearn import svm
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score


def caricamento_pulizia_dati(file_metadati, file_abbondanze, condizione_negativa='control'):
    """ output:x, y_binary, metadati_finali_no_disease, metadati_esclusi """
    # Caricamento dei dati
    metadati = pd.read_csv(file_metadati, index_col=0)
    abbondanze = pd.read_csv(file_abbondanze, index_col=0)
    # Trasponiamo il DataFrame delle abbondanze per avere i campioni come righe e le specie come colonne
    abbondanze = abbondanze.T

    # Analisi dei metadati
    # Contiamo i valori mancanti per ogni colonna
    metadati_mancanti = metadati.isnull().sum().sort_values(ascending=False)
    # Calcoliamo la percentuale di valori mancanti per ogni colonna
    percentuali_null = metadati.isnull().mean()
    # Selezioniamo solo le colonne con meno del 50% di valori mancanti
    metadati_utilizzabili = metadati.loc[:, percentuali_null < 0.5]

    # Rimuoviamo le colonne che non sono utili per l'analisi (ad esempio, quelle con molti valori mancanti o che non sono rilevanti per la classificazione)
    metadati_da_rimuovere = ["number_reads", "non_westernized", "BMI", "curator", "number_bases",
                             "minimum_read_length", "PMID", "median_read_length", "body_site", "DNA_extraction_kit"]
    # Creiamo un nuovo DataFrame con solo i metadati finali da utilizzare per l'analisi
    metadati_esclusi = metadati_utilizzabili[metadati_da_rimuovere]
    # Creiamo un nuovo DataFrame con solo i metadati finali da utilizzare per l'analisi
    metadati_finali = metadati_utilizzabili.drop(columns=metadati_da_rimuovere)

    # Preparazione dei dati per la classificazione
    y = metadati_finali["study_condition"]
    x = abbondanze

    if condizione_negativa == "control":
        y = metadati_finali["study_condition"]
        y_binary = y.map({"CRC": 1, "control": 0})
        
    elif condizione_negativa == "healthy":
        y = metadati_finali["disease"]
        y_binary = y.map({"CRC": 1, "healthy": 0})
        
    else:
        raise ValueError("Scenario non valido. Scegli 'control' o 'healthy'.")

    y_binary = y_binary.dropna()

    # Allineiamo x e y_binary per assicurarci che abbiano gli stessi campioni (righe)
    x, y_binary = x.align(y_binary, join="inner", axis=0)
    return x, y_binary, metadati_finali, metadati_esclusi

def filtro_relative(x):
    x_nuovo = x.copy()
    for feature in x_nuovo.columns:
        # Estraiamo direttamente le percentuali di questo batterio
        percentuali_feature = x_nuovo[feature]
        
        # 3. Creiamo la maschera per trovare i pazienti sotto la soglia di rumore
        pazienti_sotto_soglia = percentuali_feature < 0.01
        
        # 4. Azzeriamo le percentuali inaffidabili
        x_nuovo.loc[pazienti_sotto_soglia, feature] = 0
            
    return x_nuovo





def filtro_artefatti(x):
    prevalenza_tot = (x > 0).mean(axis=0)
    x_clean = x.loc[:, (prevalenza_tot > 0.01)] 
    return x_clean

def maschera_prevalenza(X_train, y_train, cutoff):
    """ output: batteri_da_tenere """
    # Analizziamo la prevalenza delle specie nei campioni sani e CRC
    X_train_sani = X_train[y_train == 0]
    X_train_crc = X_train[y_train == 1]

    # Calcoliamo la prevalenza di ogni specie nei campioni sani e CRC
    prevalenza_sani = (X_train_sani > 0).mean(axis=0)
    prevalenza_crc = (X_train_crc > 0).mean(axis=0)

    # Selezioniamo le specie che sono presenti in almeno il 10% dei campioni in uno dei due gruppi
    batteri_da_tenere = X_train.columns[(
        prevalenza_sani > cutoff) | (prevalenza_crc > cutoff)]
    return batteri_da_tenere


def trasformazione_clr(X):
    """output: X_filtrato"""
    X_check = X.copy()
    if (X_check.sum(axis=1) == 0).any():
        # Aggiungiamo un valore infinitesimale solo dove necessario
        X_check = X_check.replace(0, 1e-9)
    # Applichiamo la trasformazione CLR ai dati filtrati
    # La funzione multi_replace sostituisce i valori zero con un piccolo valore positivo per evitare problemi con la trasformazione CLR
    X_nozeri = multi_replace(X)
    X_clr_array = clr(X_nozeri)
    # Creiamo un DataFrame per i dati trasformati, mantenendo gli stessi indici e colonne del DataFrame originale
    X_clr = pd.DataFrame(X_clr_array, index=X.index, columns=X.columns)
    return X_clr

def filtraggio(X_clr, batteri_da_tenere):
    X_filtrato = X_clr[batteri_da_tenere].copy()
    X_filtrato.columns = X_filtrato.columns.str.replace(r"\[|\]|<", "", regex=True)
    return X_filtrato


def applica_algoritmo(modello, X_train, X_test, y_train, y_test, nome_test):
    """ output: report, matrice di confusione """
    modello.fit(X_train, y_train)
    y_pred = modello.predict(X_test)
    # Genera il report come dizionario
    report = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).T
    print(df_report)
    df_report.to_csv(f"{nome_test}.csv", index=True)
    cm = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    return df_report, cm


def standard_scaler(train, test):
    """ output: train_scaled, test_scaled """
    scaler = StandardScaler()
    train_scaled = pd.DataFrame(
        scaler.fit_transform(train),
        index=train.index,
        columns=train.columns
    )
    test_scaled = pd.DataFrame(
        scaler.transform(test),
        index=test.index,
        columns=test.columns
    )
    return train_scaled, test_scaled

def crossvalidation (modello, X, y, cv, scoring):
    scores = cross_val_score(modello, X, y, cv=cv, scoring=scoring)
    report_cv = pd.Series([scores.mean(), scores.std()], index=["Media", "Deviazione_standard"])
    #print(scores)
    print(report_cv)
    return report_cv
    