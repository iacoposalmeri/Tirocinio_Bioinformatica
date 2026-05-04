import pandas as pd
import matplotlib.pyplot as plt

# 1. Carica i metadati
metadati = pd.read_csv("Metadati_CRC_Dataset.csv", low_memory=False)

# 2. Scegli cosa esplorare (prova a cambiare questa stringa!)
colonna_da_esplorare = 'age_category'  # Alternative: 'age_category', 'country', 'BMI', 'gender'

print(f"=== DISTRIBUZIONE DELLA VARIABILE: {colonna_da_esplorare} ===")
distribuzione = metadati[colonna_da_esplorare].value_counts(dropna=False)
print(distribuzione)

# 3. Creiamo un grafico a barre veloce
distribuzione.plot(kind='bar', color='skyblue', edgecolor='black', figsize=(8, 4))
plt.title(f"Distribuzione dei pazienti per {colonna_da_esplorare}")
plt.ylabel("Numero di Pazienti")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"distribuzione_{colonna_da_esplorare}.png") # Salva l'immagine
print(f"\nGrafico salvato come 'distribuzione_{colonna_da_esplorare}.png'")