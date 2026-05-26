import pandas as pd
from sklearn.model_selection import train_test_split
from utils_crc2 import *

SEED = 42
CUTOFF = 0.2
K_FEATURE = 100

x, y_binary, _, _ = caricamento_pulizia_dati("Metadati_CRC_Dataset.csv", "Abbondanze_CRC_Dataset.csv", condizione_negativa='healthy')
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(x, y_binary, test_size=0.2, random_state=SEED, stratify=y_binary)

bacteria_to_keep = maschera_prevalenza(X_train_full, y_train_full, cutoff=CUTOFF)
X_train_filt = filtraggio(X_train_full, bacteria_to_keep)
X_test_filt = filtraggio(X_test_full, bacteria_to_keep)

filtro_vincente = ConsensusFilter(k=K_FEATURE, threshold=2)

filtro_vincente.fit(X_train_filt, y_train_full)

indici_batteri = filtro_vincente.selected_indices_

batteri_vincitori = X_test_filt.columns[indici_batteri]

print(f"\nIL CONSENSUS HA ISOLATO {len(batteri_vincitori)} SUPER-BATTERI:")
print("-" * 50)

lista_nomi = []
for i, batterio in enumerate(batteri_vincitori, 1):
    print(f"{i}. {batterio}")
    lista_nomi.append(batterio)

df_batteri = pd.DataFrame({'Biomarcatore (Tassonomia)': lista_nomi})
df_batteri.to_csv("Firma_Microbica_Vincente_CRC.csv", index=False)
print("\nSalvati in 'Firma_Microbica_Vincente_CRC_MAGGIORANZA.csv'")