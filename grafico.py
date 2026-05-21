import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# ==========================================
# PANNELLO DI CONTROLLO (MODULARE)
# ==========================================
tipo_riduzione = "no_riduzioni"  # Cambia in "SKB", "PCA" o "no_riduzioni"

if tipo_riduzione == "PCA":
    file_input = 'risultati_crossvalidation_pca.csv'
    colonna_x = 'varianza_pca'          
    label_x = 'Varianza Spiegata (PCA)'
    suffisso_salvataggio = 'pca'
    
elif tipo_riduzione == "SKB":
    file_input = 'risultati_RCLR_crossvalidation_skb.csv'
    colonna_x = 'k_skb'
    label_x = 'Numero di feature (k_skb)'
    suffisso_salvataggio = 'skb'
    
elif tipo_riduzione == "no_riduzioni":
    file_input = 'risultati_crossvalidation_no_riduzioni_rclr.csv' # <-- INSERISCI QUI IL NOME DEL TUO CSV SENZA RIDUZIONI
    colonna_x = 'riduzione'
    label_x = 'Riduzione Dimensionalità'
    suffisso_salvataggio = 'no_riduzioni'

# 1. Caricamento UNICO della tabella dinamica
df = pd.read_csv(file_input)

# TRUCCO GENIALE PER "no_riduzioni": 
# Creiamo una colonna finta per dare un asse X alla Heatmap
if tipo_riduzione == "no_riduzioni":
    df['riduzione'] = "Nessuna (Tutti i batteri)"

# Filtriamo il dataframe globale per avere solo i modelli che ci interessano
df_modelli = df[df['Modello'].isin(['Random Forest', 'XGB', 'SVM'])]

""" 
# ==========================================
# PARTE 1: GRAFICI SINGOLI (MEDIA F1 MACRO)
# ==========================================

# --- RANDOM FOREST ---
df_RF = df[df['Modello'] == 'Random Forest']

griglia = sns.FacetGrid(df_RF, col="cutoff", col_wrap=3, height=4)
griglia.map(plt.errorbar, colonna_x, "Media_f1_macro", "Deviazione_standard_f1_macro", marker="o", capsize=4)
griglia.set_axis_labels(label_x, "Media F1 Macro")
griglia.figure.suptitle(f'Random Forest: Media F1 Macro vs {label_x} per ogni Cutoff', fontweight='bold', fontsize=14)
griglia.figure.subplots_adjust(top=0.9)
plt.savefig(f'grafico_rf_asseX_{suffisso_salvataggio}.png', bbox_inches='tight')  
plt.show()

griglia = sns.FacetGrid(df_RF, col=colonna_x, col_wrap=5, height=4)
griglia.map(plt.errorbar, "cutoff", "Media_f1_macro", "Deviazione_standard_f1_macro", marker="o", capsize=4)
griglia.set_axis_labels("Cutoff", "Media F1 Macro")
griglia.figure.suptitle(f'Random Forest: Media F1 Macro vs Cutoff per ogni {label_x}', fontweight='bold', fontsize=14)
griglia.figure.subplots_adjust(top=0.9)
plt.savefig(f'grafico_rf_cutoff_{suffisso_salvataggio}.png', bbox_inches='tight')  
plt.show()

# --- XGBOOST ---
df_XGB = df[df['Modello'] == 'XGB']

griglia = sns.FacetGrid(df_XGB, col="cutoff", col_wrap=3, height=4)
griglia.map(plt.errorbar, colonna_x, "Media_f1_macro", "Deviazione_standard_f1_macro", marker="o", capsize=4)
griglia.set_axis_labels(label_x, "Media F1 Macro")
griglia.figure.suptitle(f'XGB: Media F1 Macro vs {label_x} per ogni Cutoff', fontweight='bold', fontsize=14)
griglia.figure.subplots_adjust(top=0.9)
plt.savefig(f'grafico_xgb_asseX_{suffisso_salvataggio}.png', bbox_inches='tight')  
plt.show()

griglia = sns.FacetGrid(df_XGB, col=colonna_x, col_wrap=5, height=4)
griglia.map(plt.errorbar, "cutoff", "Media_f1_macro", "Deviazione_standard_f1_macro", marker="o", capsize=4)
griglia.set_axis_labels("Cutoff", "Media F1 Macro")
griglia.figure.suptitle(f'XGB: Media F1 Macro vs Cutoff per ogni {label_x}', fontweight='bold', fontsize=14)
griglia.figure.subplots_adjust(top=0.9)
plt.savefig(f'grafico_xgb_cutoff_{suffisso_salvataggio}.png', bbox_inches='tight')  
plt.show()

# --- SVM ---
df_SVM = df[df['Modello'] == 'SVM']

griglia = sns.FacetGrid(df_SVM, col="cutoff", col_wrap=3, height=4)
griglia.map(plt.errorbar, colonna_x, "Media_f1_macro", "Deviazione_standard_f1_macro", marker="o", capsize=4)
griglia.set_axis_labels(label_x, "Media F1 Macro")
griglia.figure.suptitle(f'SVM: Media F1 Macro vs {label_x} per ogni Cutoff', fontweight='bold', fontsize=14)
griglia.figure.subplots_adjust(top=0.9)
plt.savefig(f'grafico_svm_asseX_{suffisso_salvataggio}.png', bbox_inches='tight')  
plt.show()

griglia = sns.FacetGrid(df_SVM, col=colonna_x, col_wrap=5, height=4)
griglia.map(plt.errorbar, "cutoff", "Media_f1_macro", "Deviazione_standard_f1_macro", marker="o", capsize=4)
griglia.set_axis_labels("Cutoff", "Media F1 Macro")
griglia.figure.suptitle(f'SVM: Media F1 Macro vs Cutoff per ogni {label_x}', fontweight='bold', fontsize=14)
griglia.figure.subplots_adjust(top=0.9)
plt.savefig(f'grafico_svm_cutoff_{suffisso_salvataggio}.png', bbox_inches='tight')  
plt.show()


# ==========================================
# PARTE 2: GRAFICI SINGOLI (DEVIAZIONE STANDARD)
# ==========================================

# --- RANDOM FOREST DS ---
griglia = sns.FacetGrid(df_RF, col="cutoff", col_wrap=3, height=4)
griglia.map(plt.plot, colonna_x, "Deviazione_standard_f1_macro", marker="o")
griglia.set_axis_labels(label_x, "Deviazione Standard F1 Macro")
griglia.figure.suptitle(f'Random Forest: Deviazione Standard vs {label_x} per ogni Cutoff', fontweight='bold', fontsize=14)
griglia.figure.subplots_adjust(top=0.9)
plt.savefig(f'grafico_rf_ds_asseX_{suffisso_salvataggio}.png', bbox_inches='tight')  
plt.show()

griglia = sns.FacetGrid(df_RF, col=colonna_x, col_wrap=5, height=4)
griglia.map(plt.plot, "cutoff", "Deviazione_standard_f1_macro", marker="o")
griglia.set_axis_labels("Cutoff", "Deviazione Standard F1 Macro")
griglia.figure.suptitle(f'Random Forest: Deviazione Standard vs Cutoff per ogni {label_x}', fontweight='bold', fontsize=14)
griglia.figure.subplots_adjust(top=0.9)
plt.savefig(f'grafico_rf_ds_cutoff_{suffisso_salvataggio}.png', bbox_inches='tight')  
plt.show()

# --- XGBOOST DS ---
griglia = sns.FacetGrid(df_XGB, col="cutoff", col_wrap=3, height=4)
griglia.map(plt.plot, colonna_x, "Deviazione_standard_f1_macro", marker="o")
griglia.set_axis_labels(label_x, "Deviazione Standard F1 Macro")
griglia.figure.suptitle(f'XGB: Deviazione Standard vs {label_x} per ogni Cutoff', fontweight='bold', fontsize=14)
griglia.figure.subplots_adjust(top=0.9)
plt.savefig(f'grafico_xgb_ds_asseX_{suffisso_salvataggio}.png', bbox_inches='tight')  
plt.show()

griglia = sns.FacetGrid(df_XGB, col=colonna_x, col_wrap=5, height=4)
griglia.map(plt.plot, "cutoff", "Deviazione_standard_f1_macro", marker="o")
griglia.set_axis_labels("Cutoff", "Deviazione Standard F1 Macro")
griglia.figure.suptitle(f'XGB: Deviazione Standard vs Cutoff per ogni {label_x}', fontweight='bold', fontsize=14)
griglia.figure.subplots_adjust(top=0.9)
plt.savefig(f'grafico_xgb_ds_cutoff_{suffisso_salvataggio}.png', bbox_inches='tight')  
plt.show()

# --- SVM DS ---
griglia = sns.FacetGrid(df_SVM, col="cutoff", col_wrap=3, height=4)
griglia.map(plt.plot, colonna_x, "Deviazione_standard_f1_macro", marker="o")
griglia.set_axis_labels(label_x, "Deviazione Standard F1 Macro")
griglia.figure.suptitle(f'SVM: Deviazione Standard vs {label_x} per ogni Cutoff', fontweight='bold', fontsize=14)
griglia.figure.subplots_adjust(top=0.9)
plt.savefig(f'grafico_svm_ds_asseX_{suffisso_salvataggio}.png', bbox_inches='tight')  
plt.show()

griglia = sns.FacetGrid(df_SVM, col=colonna_x, col_wrap=5, height=4)
griglia.map(plt.plot, "cutoff", "Deviazione_standard_f1_macro", marker="o")
griglia.set_axis_labels("Cutoff", "Deviazione Standard F1 Macro")
griglia.figure.suptitle(f'SVM: Deviazione Standard vs Cutoff per ogni {label_x}', fontweight='bold', fontsize=14)
griglia.figure.subplots_adjust(top=0.9)
plt.savefig(f'grafico_svm_ds_cutoff_{suffisso_salvataggio}.png', bbox_inches='tight')  
plt.show()


# ==========================================
# PARTE 3: GRAFICI COMPARATIVI PER TUTTI I MODELLI
# ==========================================

# --- 1A: Media F1 Macro vs Asse X ---
g1 = sns.catplot(
    data=df_modelli, 
    kind="point", 
    x=colonna_x, 
    y="Media_f1_macro", 
    hue="Modello", 
    col="cutoff", 
    col_wrap=3, 
    height=4, 
    markers=["o", "s", "D"],
    dodge=True 
)
g1.set_axis_labels(label_x, "Media F1 Macro")
g1.figure.suptitle(f'Confronto Modelli: Media F1 Macro vs {label_x} per ogni Cutoff', fontweight='bold', fontsize=14)
g1.figure.subplots_adjust(top=0.9)
plt.savefig(f'confronto_modelli_media_asseX_{suffisso_salvataggio}.png', bbox_inches='tight')
plt.show()

# --- 1B: Media F1 Macro vs Cutoff ---
g2 = sns.catplot(
    data=df_modelli, 
    kind="point", 
    x="cutoff", 
    y="Media_f1_macro", 
    hue="Modello", 
    col=colonna_x, 
    col_wrap=5, 
    height=4, 
    markers=["o", "s", "D"],
    dodge=True
)
g2.set_axis_labels("Cutoff", "Media F1 Macro")
g2.figure.suptitle(f'Confronto Modelli: Media F1 Macro vs Cutoff per ogni {label_x}', fontweight='bold', fontsize=14)
g2.figure.subplots_adjust(top=0.9)
plt.savefig(f'confronto_modelli_media_cutoff_{suffisso_salvataggio}.png', bbox_inches='tight')
plt.show()

# --- 2A: Deviazione Standard vs Asse X ---
g3 = sns.catplot(
    data=df_modelli, 
    kind="point", 
    x=colonna_x, 
    y="Deviazione_standard_f1_macro", 
    hue="Modello", 
    col="cutoff", 
    col_wrap=3, 
    height=4, 
    markers=["o", "s", "D"],
    dodge=True
)
g3.set_axis_labels(label_x, "Deviazione Standard F1 Macro")
g3.figure.suptitle(f'Confronto Modelli: Deviazione Standard vs {label_x} per ogni Cutoff', fontweight='bold', fontsize=14)
g3.figure.subplots_adjust(top=0.9)
plt.savefig(f'confronto_modelli_ds_asseX_{suffisso_salvataggio}.png', bbox_inches='tight')
plt.show()

# --- 2B: Deviazione Standard vs Cutoff ---
g4 = sns.catplot(
    data=df_modelli, 
    kind="point", 
    x="cutoff", 
    y="Deviazione_standard_f1_macro", 
    hue="Modello", 
    col=colonna_x, 
    col_wrap=5, 
    height=4, 
    markers=["o", "s", "D"],
    dodge=True
)
g4.set_axis_labels("Cutoff", "Deviazione Standard F1 Macro")
g4.figure.suptitle(f'Confronto Modelli: Deviazione Standard vs Cutoff per ogni {label_x}', fontweight='bold', fontsize=14)
g4.figure.subplots_adjust(top=0.9)
plt.savefig(f'confronto_modelli_ds_cutoff_{suffisso_salvataggio}.png', bbox_inches='tight')
plt.show()
 """



modelli = ['Random Forest', 'XGB', 'SVM']

# --- HEATMAP MEDIA F1 MACRO ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

# 1. Calcoliamo il Min e il Max GLOBALI per la Media F1
vmin_media = df_modelli['Media_f1_macro'].min()
vmax_media = df_modelli['Media_f1_macro'].max()

for i, modello in enumerate(modelli):
    df_temp = df_modelli[df_modelli['Modello'] == modello]
    matrice = df_temp.pivot_table(index='cutoff', columns=colonna_x, values='Media_f1_macro')
    matrice = matrice.sort_index(ascending=False)
    
    # Creazione delle etichette personalizzate (F1 + n. componenti se è PCA)
    annot_labels = np.empty_like(matrice, dtype=object)
    
    if tipo_riduzione == "PCA" and 'n_componenti' in df_temp.columns:
        matrice_n = df_temp.pivot_table(index='cutoff', columns=colonna_x, values='n_componenti')
        matrice_n = matrice_n.sort_index(ascending=False)
        
        for r in range(matrice.shape[0]):
            for c in range(matrice.shape[1]):
                val_f1 = matrice.iloc[r, c]
                val_n = matrice_n.iloc[r, c]
                if pd.notna(val_f1):
                    annot_labels[r, c] = f"{val_f1:.3f}\n(n={int(val_n)})"
                else:
                    annot_labels[r, c] = ""
    else:
        # Fallback se sei in modalità SKB (mostra solo il valore F1)
        for r in range(matrice.shape[0]):
            for c in range(matrice.shape[1]):
                val_f1 = matrice.iloc[r, c]
                if pd.notna(val_f1):
                    annot_labels[r, c] = f"{val_f1:.3f}"
                else:
                    annot_labels[r, c] = ""
    
    sns.heatmap(
        matrice, 
        ax=axes[i], 
        annot=annot_labels, # <-- Usiamo le nostre etichette custom!
        fmt="",             # <-- Fondamentale lasciare vuoto quando si usano stringhe custom
        cmap="YlGnBu",    
        cbar=(i == 2),    
        linewidths=.5,
        vmin=vmin_media,  
        vmax=vmax_media   
    )
    
    axes[i].set_title(modello, fontweight='bold', fontsize=12)
    axes[i].set_ylabel('Cutoff')
    axes[i].set_xlabel(label_x)

fig.suptitle(f'Mappe di Calore: Media F1 Macro (Cutoff vs {label_x})', fontweight='bold', fontsize=16)
plt.tight_layout()
fig.subplots_adjust(top=0.88)
plt.savefig(f'heatmap_modelli_media_{suffisso_salvataggio}_rclr.png', dpi=300, bbox_inches='tight')
plt.show()


# --- HEATMAP DEVIAZIONE STANDARD ---
fig_ds, axes_ds = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

# 2. Calcoliamo il Min e il Max GLOBALI per la Deviazione Standard
vmin_ds = df_modelli['Deviazione_standard_f1_macro'].min()
vmax_ds = df_modelli['Deviazione_standard_f1_macro'].max()

for i, modello in enumerate(modelli):
    df_temp = df_modelli[df_modelli['Modello'] == modello]
    matrice_ds = df_temp.pivot_table(index='cutoff', columns=colonna_x, values='Deviazione_standard_f1_macro')
    matrice_ds = matrice_ds.sort_index(ascending=False)
    
    # Creazione delle etichette personalizzate per la deviazione standard
    annot_labels_ds = np.empty_like(matrice_ds, dtype=object)
    
    if tipo_riduzione == "PCA" and 'n_componenti' in df_temp.columns:
        matrice_n = df_temp.pivot_table(index='cutoff', columns=colonna_x, values='n_componenti')
        matrice_n = matrice_n.sort_index(ascending=False)
        
        for r in range(matrice_ds.shape[0]):
            for c in range(matrice_ds.shape[1]):
                val_ds = matrice_ds.iloc[r, c]
                val_n = matrice_n.iloc[r, c]
                if pd.notna(val_ds):
                    annot_labels_ds[r, c] = f"{val_ds:.3f}\n(n={int(val_n)})"
                else:
                    annot_labels_ds[r, c] = ""
    else:
        for r in range(matrice_ds.shape[0]):
            for c in range(matrice_ds.shape[1]):
                val_ds = matrice_ds.iloc[r, c]
                if pd.notna(val_ds):
                    annot_labels_ds[r, c] = f"{val_ds:.3f}"
                else:
                    annot_labels_ds[r, c] = ""
    
    sns.heatmap(
        matrice_ds, 
        ax=axes_ds[i], 
        annot=annot_labels_ds, # <-- Etichette custom anche qui
        fmt="",                # <-- Lasciare vuoto
        cmap="OrRd",     
        cbar=(i == 2),
        linewidths=.5,
        vmin=vmin_ds,     
        vmax=vmax_ds      
    )
    
    axes_ds[i].set_title(modello, fontweight='bold', fontsize=12)
    axes_ds[i].set_ylabel('Cutoff')
    axes_ds[i].set_xlabel(label_x)

fig_ds.suptitle(f'Mappe di Calore: Deviazione Standard F1 Macro (Cutoff vs {label_x})', fontweight='bold', fontsize=16)
plt.tight_layout()
fig_ds.subplots_adjust(top=0.88)
plt.savefig(f'heatmap_modelli_ds_{suffisso_salvataggio}_rclr.png', dpi=300, bbox_inches='tight')
plt.show()