from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from xgboost import XGBClassifier
from utils_crc2 import *
cutoff = 0.2

x, y_binary, metadati_finali_no_desease, metadati_esclusi = caricamento_pulizia_dati("Metadati_CRC_Dataset.csv", "Abbondanze_CRC_Dataset.csv")



X_train, X_test, y_train, y_test = train_test_split(x, y_binary, test_size=0.2, random_state=42, stratify=y_binary)


batteri_da_tenere = maschera_prevalenza(X_train, y_train, cutoff)

X_train_rclr = trasformazione_rclr_nativa(X_train)
X_test_rclr = trasformazione_rclr_nativa(X_test)
print(f"Prima del filtraggio al {cutoff*100:.0f}%: {X_train_rclr.shape}")
X_train_filtrato = filtraggio(X_train_rclr, batteri_da_tenere)
X_test_filtrato = filtraggio(X_test_rclr, batteri_da_tenere)
print(f"Dopo il filtraggio al {cutoff*100:.0f}%: {X_train_filtrato.shape}")