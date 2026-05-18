import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import svm
from sklearn.metrics import classification_report
from funzioni_crc import maschera_prevalenza, caricamento_pulizia_dati, filtraggio, trasformazione_clr, standard_scaler, pca_classica

# ==========================================
# 1. SETUP DELLA PIPELINE VINCITRICE
# ==========================================
cutoff = 0.05

print("Caricamento dati...")
x, y_binary, metadati_finali_no_desease, metadati_esclusi = caricamento_pulizia_dati("Metadati_CRC_Dataset.csv", "Abbondanze_CRC_Dataset.csv")

# Split e CLR
X_train, X_test, y_train, y_test = train_test_split(x, y_binary, test_size=0.2, random_state=42, stratify=y_binary)
X_train_clr = trasformazione_clr(X_train)
X_test_clr = trasformazione_clr(X_test)

# Filtraggio Prevalenza
batteri_da_tenere = maschera_prevalenza(X_train, y_train, cutoff)
print(f"Prima del filtraggio al {cutoff*100:.0f}%: {X_train_clr.shape}")

X_train_filtrato = filtraggio(X_train_clr, batteri_da_tenere)
X_test_filtrato = filtraggio(X_test_clr, batteri_da_tenere)
print(f"Dopo il filtraggio al {cutoff*100:.0f}%: {X_train_filtrato.shape}")

# Scaling 
X_train_scaled, X_test_scaled = standard_scaler(X_train_filtrato, X_test_filtrato)

# ==========================================
# 2. IMPOSTAZIONE GRID SEARCH CV
# ==========================================
# Griglia dei parametri per l'SVM
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100, 1000],
    'gamma': ['scale', 'auto', 0.0001, 0.001, 0.01, 0.1, 1],
    'kernel': ['rbf']
}

SVM_base = svm.SVC(random_state=42)

# Configurazione della ricerca (cv=10 fold, ottimizzazione per f1_macro)
grid_search = GridSearchCV(
    estimator=SVM_base, 
    param_grid=param_grid, 
    cv=10, 
    scoring='f1_macro', 
    n_jobs=-1,      # Usa tutti i core del processore
    verbose=2       # Stampa un log per mostrare l'avanzamento
)

# ==========================================
# 3. ESECUZIONE E RISULTATI
# ==========================================
print("\nInizio Grid Search su SVM. Questo processo potrebbe richiedere alcuni minuti...")
grid_search.fit(X_train_scaled, y_train)

# --- STAMPE FONDAMENTALI AGGIUNTE ---
print("\n" + "="*50)
print("🏆 RISULTATI GRID SEARCH (Cross-Validation su Train Set)")
print("="*50)
print(f"Migliori Parametri Trovati: {grid_search.best_params_}")
print(f"Miglior F1-Macro Medio: {grid_search.best_score_:.4f}")
# ------------------------------------

# Estrazione del modello "vincitore"
miglior_modello = grid_search.best_estimator_

# Valutazione finale sul TEST SET (dati mai visti prima)
y_pred_test = miglior_modello.predict(X_test_scaled)

print("\n" + "="*50)
print("📊 PERFORMANCE FINALE SUL TEST SET (Dati Invisibili)")
print("="*50)
print(classification_report(y_test, y_pred_test))