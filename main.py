from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from xgboost import XGBClassifier
from funzioni_crc import filtro_relative, maschera_prevalenza, caricamento_pulizia_dati, filtraggio, trasformazione_clr, applica_algoritmo, standard_scaler, crossvalidation, filtro_artefatti
cutoffs = [0.03, 0.05, 0.07, 0.1, 0.15, 0.2]

x, y_binary, metadati_finali_no_desease, metadati_esclusi = caricamento_pulizia_dati("Metadati_CRC_Dataset.csv", "Abbondanze_CRC_Dataset.csv")



X_train, X_test, y_train, y_test = train_test_split(x, y_binary, test_size=0.2, random_state=42, stratify=y_binary)

for cutoff in cutoffs:

    batteri_da_tenere = maschera_prevalenza(X_train, y_train, cutoff)

    X_train_clr = trasformazione_clr(X_train)
    X_test_clr = trasformazione_clr(X_test)
    print(f"Prima del filtraggio al {cutoff*100:.0f}%: {X_train_clr.shape}")
    X_train_filtrato = filtraggio(X_train_clr, batteri_da_tenere)
    X_test_filtrato = filtraggio(X_test_clr, batteri_da_tenere)
    X_train_scaled, X_test_scaled = standard_scaler(X_train_filtrato, X_test_filtrato)
    print(f"Dopo il filtraggio al {cutoff*100:.0f}%: {X_train_filtrato.shape}")


    print(f"Random Forest {cutoff*100:.0f}%:")
    RF = RandomForestClassifier(random_state=42, n_jobs=-1)
    report_cv_rf = crossvalidation(RF, X_train_filtrato, y_train, 10, "f1_macro")

    print(f"XGB {cutoff*100:.0f}%:")
    XGB = XGBClassifier(n_jobs=-1)
    report_cv_xgb = crossvalidation(XGB, X_train_filtrato, y_train, 10, "f1_macro")

    print(f"SVM {cutoff*100:.0f}%:")
    SVM = svm.SVC(kernel="rbf", random_state=42)
    report_cv_svm = crossvalidation(SVM, X_train_scaled, y_train, 10, "f1_macro")
