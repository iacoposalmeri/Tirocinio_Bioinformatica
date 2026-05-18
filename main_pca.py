import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from xgboost import XGBClassifier
from funzioni_crc import maschera_prevalenza, caricamento_pulizia_dati, filtraggio, trasformazione_clr, standard_scaler, crossvalidation, pca_classica, pca_grafico, pca_grafico2
from sklearn.feature_selection import SelectKBest, mutual_info_classif
cutoffs = [0.03, 0.05, 0.07, 0.1, 0.15, 0.2]
varianze_pca = [0.7, 0.8, 0.9, 0.95] #varianze da testare per la PCA

x, y_binary, metadati_finali_no_desease, metadati_esclusi = caricamento_pulizia_dati("Metadati_CRC_Dataset.csv", "Abbondanze_CRC_Dataset.csv")

X_train, X_test, y_train, y_test = train_test_split(x, y_binary, test_size=0.2, random_state=42, stratify=y_binary)
X_train_clr = trasformazione_clr(X_train)
X_test_clr = trasformazione_clr(X_test)

risultati_cv = {"cutoff": [], "varianza_pca": [],  "n_componenti": [], "Modello": [], "Media_f1_macro": [], "Deviazione_standard_f1_macro": []}

for cutoff in cutoffs:
        batteri_da_tenere = maschera_prevalenza(X_train, y_train, cutoff)

        print(f"Prima del filtraggio al {cutoff*100:.0f}%: {X_train_clr.shape}")
        X_train_filtrato = filtraggio(X_train_clr, batteri_da_tenere)
        X_test_filtrato = filtraggio(X_test_clr, batteri_da_tenere)
        X_train_scaled, X_test_scaled = standard_scaler(X_train_filtrato, X_test_filtrato)


        for varianza in varianze_pca:
                        X_train_pca_scaled, X_test_pca_scaled, pca_fitted_scaled = pca_classica(X_train_scaled, X_test_scaled, n_components=varianza)
                        num_componenti_scaled = pca_fitted_scaled.n_components_
                        
                        X_train_pca, X_test_pca, pca_fitted = pca_classica(X_train_filtrato, X_test_filtrato, n_components=varianza)
                        num_componenti = pca_fitted.n_components_
                        print(f"Random Forest {cutoff*100:.0f}%:")

                        RF = RandomForestClassifier(random_state=42, n_jobs=-1)
                        report_cv_rf = crossvalidation(RF, X_train_pca, y_train, 10, "f1_macro")
                        risultati_cv["cutoff"].append(cutoff)
                        risultati_cv["varianza_pca"].append(varianza)
                        risultati_cv["n_componenti"].append(num_componenti)
                        risultati_cv["Modello"].append("Random Forest")
                        risultati_cv["Media_f1_macro"].append(report_cv_rf.iloc[0])
                        risultati_cv["Deviazione_standard_f1_macro"].append(report_cv_rf.iloc[1])

                        print(f"XGB {cutoff*100:.0f}%:")
                        XGB = XGBClassifier(n_jobs=-1)
                        report_cv_xgb = crossvalidation(XGB, X_train_pca, y_train, 10, "f1_macro")
                        risultati_cv["cutoff"].append(cutoff)
                        risultati_cv["varianza_pca"].append(varianza)
                        risultati_cv["n_componenti"].append(num_componenti)
                        risultati_cv["Modello"].append("XGB")
                        risultati_cv["Media_f1_macro"].append(report_cv_xgb.iloc[0])
                        risultati_cv["Deviazione_standard_f1_macro"].append(report_cv_xgb.iloc[1])

                        print(f"SVM {cutoff*100:.0f}%:")
                        SVM = svm.SVC(kernel="rbf", random_state=42)
                        report_cv_svm = crossvalidation(SVM, X_train_pca_scaled, y_train, 10, "f1_macro")
                        risultati_cv["cutoff"].append(cutoff)
                        risultati_cv["varianza_pca"].append(varianza)
                        risultati_cv["n_componenti"].append(num_componenti_scaled)
                        risultati_cv["Modello"].append("SVM")
                        risultati_cv["Media_f1_macro"].append(report_cv_svm.iloc[0])
                        risultati_cv["Deviazione_standard_f1_macro"].append(report_cv_svm.iloc[1])

df_risultati_cv = pd.DataFrame(risultati_cv)
df_risultati_cv.to_csv("risultati_crossvalidation_pca.csv", index=False)                        