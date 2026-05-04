import pandas as pd

# 1. Carica le abbondanze
abbondanze = pd.read_csv("Abbondanze_CRC_Dataset.csv", low_memory=False)
abbondanze = abbondanze.rename(columns={'Unnamed: 0': 'Taxonomy'})

# 2. Estraiamo solo i dati numerici
batteri = abbondanze['Taxonomy']
dati_numerici = abbondanze.drop(columns=['Taxonomy'])

print("=== TOP 10 BATTERI PIÙ ABBONDANTI (Media globale) ===")
# Calcoliamo la media per ogni riga (batterio) lungo tutte le colonne (pazienti)
medie_batteri = dati_numerici.mean(axis=1)

# Uniamo i nomi dei batteri alle loro medie
classifica = pd.DataFrame({
    'Batterio': batteri,
    'Abbondanza_Media_Globale': medie_batteri
})

# Ordiniamo in modo decrescente e prendiamo i primi 10
top_10 = classifica.sort_values(by='Abbondanza_Media_Globale', ascending=False).head(10)

# Stampiamo il risultato in modo pulito
for index, row in top_10.iterrows():
    nome_pulito = row['Batterio'].split(":")[-1] # Toglie la scritta "species:" per pulizia
    print(f"- {nome_pulito}: {row['Abbondanza_Media_Globale']:.2f}%")