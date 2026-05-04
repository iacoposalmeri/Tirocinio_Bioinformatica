import pandas as pd

# 1. Carica dati
abbondanze = pd.read_csv("Abbondanze_CRC_Dataset.csv", low_memory=False)
dati_numerici = abbondanze.drop(columns=['Unnamed: 0'])

print("=== ANALISI DI PREVALENZA E SPARSITÀ (Feature Selection) ===")

# Calcoliamo in quanti pazienti (colonne) è presente ogni batterio (> 0)
prevalenza = (dati_numerici > 0).sum(axis=1)
percentuale_prevalenza = (prevalenza / dati_numerici.shape[1]) * 100

# Quanti batteri sono "rarissimi" (presenti in meno del 5% dei pazienti)?
soglia_rari = 5.0
batteri_rari = (percentuale_prevalenza < soglia_rari).sum()

print(f"Totale specie batteriche: {dati_numerici.shape[0]}")
print(f"Specie rarissime (< {soglia_rari}% dei pazienti): {batteri_rari}")
print(f"Percentuale di feature che probabilmente andranno scartate: {(batteri_rari/dati_numerici.shape[0])*100:.1f}%")

print("\n(Questo è un dato utilissimo per la tua tesi: la riduzione della dimensionalità")
print("inizia spesso tagliando le feature che compaiono quasi esclusivamente come zeri!)")