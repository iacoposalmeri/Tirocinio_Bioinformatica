import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from skbio.stats.composition import clr, multi_replace
from sklearn import svm
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
#from gemelli.preprocessing import matrix_rclr, rpca
#from gemelli.rpca import rpca
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def pca_classica(X_train, X_test, n_components):
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
   
    return X_train_pca, X_test_pca, pca
    
def pca_grafico(pca):
    PC_values = np.arange(pca.n_components_) + 1
    plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained')
    plt.show()
def pca_grafico2(pca_modello, cutoff, varianza):
    '''
    Genera e salva lo Scree Plot mostrando varianza singola, cumulata 
    e le linee di soglia per numero di componenti e target.
    '''
    # 1. Numero di componenti estratte
    numero_reale_componenti = len(pca_modello.explained_variance_ratio_)
    PC_values = np.arange(numero_reale_componenti) + 1
    
    # 2. Calcolo Varianza Singola e Cumulata (in percentuale)
    varianza_singola = pca_modello.explained_variance_ratio_ * 100
    varianza_cumulata = np.cumsum(varianza_singola)
    
    # Inizializza la figura (un po' più larga per far spazio alla legenda)
    plt.figure(figsize=(10, 6))
    
    # 3. Disegna le barre per la singola e la linea per la cumulata
    plt.bar(PC_values, varianza_singola, alpha=0.5, color='royalblue', label='Varianza Singola')
    plt.plot(PC_values, varianza_cumulata, 'o-', linewidth=2, color='darkorange', label='Varianza Cumulata')
    
    # 4. Aggiungi il "mirino" (Soglia Varianza e Numero Componenti)
    target_perc = varianza * 100
    plt.axhline(y=target_perc, color='red', linestyle='--', alpha=0.8, 
                label=f'Target Varianza ({target_perc:.0f}%)')
    
    plt.axvline(x=numero_reale_componenti, color='green', linestyle='--', alpha=0.8, 
                label=f'N. Componenti Usate ({numero_reale_componenti})')
    
    # 5. Estetica e descrizioni
    plt.title(f'Scree Plot (Cutoff Prevalenza: {cutoff*100:.0f}%)')
    plt.xlabel('Numero della Componente Principale (PC)')
    plt.ylabel('Varianza Spiegata (%)')
    
    # Mostra la legenda con tutti i dettagli
    plt.legend(loc='lower right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 6. Salvataggio
    nome_file = f"ScreePlot_Cut_0{str(cutoff).split('.')[1]}_Var_{varianza}.png"
    plt.savefig(nome_file, dpi=300)
    plt.close()
    

def caricamento_pulizia_dati(file_metadati, file_abbondanze):
    '''output:x, y_binary, metadati_finali_no_disease, metadati_esclusi'''
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

    # Creiamo un nuovo DataFrame che contiene solo i campioni con condizioni di studio "CRC" o "healthy"
    metadati_finali_no_disease = metadati_finali[metadati_finali["disease"].isin([
                                                                                 "CRC", "healthy"])]

    # Preparazione dei dati per la classificazione
    y = metadati_finali_no_disease["study_condition"]
    x = abbondanze

    # Convertiamo le etichette in valori binari (CRC = 1, control = 0)
    y_binary = y.map({"CRC": 1, "control": 0})

    # Allineiamo x e y_binary per assicurarci che abbiano gli stessi campioni (righe)
    x, y_binary = x.align(y_binary, join="inner", axis=0)
    return x, y_binary, metadati_finali_no_disease, metadati_esclusi

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





def trasformazione_rclr_gemelli(X):
    '''
    Applica la Robust CLR usando gemelli. 
    Questa trasformazione lavora riga per riga (campione per campione).
    '''
    # matrix_rclr accetta array NumPy (o tabelle BIOM), quindi passiamo X.values
    # Restituisce un array NumPy con i valori trasformati
    X_rclr_array = matrix_rclr(X.values)
    
    # Ricostruiamo il DataFrame mantenendo indici e nomi dei batteri
    X_rclr = pd.DataFrame(X_rclr_array, index=X.index, columns=X.columns)
    
    return X_rclr

def riduzione_rpca_gemelli(X_train, X_test, n_components=5):
    # Passiamo direttamente i DataFrame
    ordination_train, _ = rpca(X_train, n_components=n_components)
    ordination_test, _ = rpca(X_test, n_components=n_components)
    
    X_train_rpca = ordination_train.samples
    X_test_rpca = ordination_test.samples
    
    return X_train_rpca, X_test_rpca

def filtro_artefatti(x):
    prevalenza_tot = (x > 0).mean(axis=0)
    x_clean = x.loc[:, (prevalenza_tot > 0.01)] 
    return x_clean

def maschera_prevalenza(X_train, y_train, cutoff):
    '''output: batteri_da_tenere'''
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
    '''output: X_filtrato'''
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
    '''output: report, matrice di confusione'''
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
    '''output: train_scaled, test_scaled'''
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
    scores = cross_val_score(modello, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    report_cv = pd.Series([scores.mean(), scores.std()], index=["Media", "Deviazione_standard"])
    #print(scores)
    print(report_cv)
    return report_cv
    