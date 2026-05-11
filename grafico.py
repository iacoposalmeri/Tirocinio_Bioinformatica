import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Carichiamo la tabella
df = pd.read_csv('risultati_crossvalidation_skb.csv')

# 2. Filtriamo per il modello desiderato
df_RF = df[df['Modello'] == 'Random Forest']

# 3. Creiamo la griglia vuota (un grafico per ogni cutoff)
griglia = sns.FacetGrid(df_RF, col="cutoff", col_wrap=3, height=4)

# 4. Disegniamo i dati (k_skb va in automatico sull'asse X!)
griglia.map(plt.errorbar, "k_skb", "Media_f1_macro", "Deviazione_standard_f1_macro", marker="o", capsize=4)

# 5. AGGIUNTA: Rimettiamo le etichette per far capire a chi legge cosa stiamo guardando
griglia.set_axis_labels("Numero di feature (k_skb)", "Media F1 Macro")
# Usiamo suptitle per il titolo globale della figura
griglia.figure.suptitle(
    'Random Forest: Media F1 Macro vs Numero di feature (k_skb) per ogni Cutoff', 
    fontweight='bold', 
    fontsize=14
)

# Regoliamo lo spazio in alto per non far sovrapporre il titolo ai grafici
griglia.figure.subplots_adjust(top=0.9)
plt.savefig('grafico_rf_k_skb.png')  # Salva il grafico come immagine
plt.show()


# 3. Creiamo la griglia vuota (un grafico per ogni cutoff)
griglia = sns.FacetGrid(df_RF, col="k_skb", col_wrap=5, height=4)

# 4. Disegniamo i dati (k_skb va in automatico sull'asse X!)
griglia.map(plt.errorbar, "cutoff", "Media_f1_macro", "Deviazione_standard_f1_macro", marker="o", capsize=4)

# 5. AGGIUNTA: Rimettiamo le etichette per far capire a chi legge cosa stiamo guardando
griglia.set_axis_labels("Cutoff", "Media F1 Macro")
griglia.figure.suptitle(
    'Random Forest: Media F1 Macro vs Cutoff per ogni Numero di feature (k_skb)',
    fontweight='bold',
    fontsize=14
)
griglia.figure.subplots_adjust(top=0.9)
plt.savefig('grafico_rf_cutoff.png')  # Salva il grafico come immagine
plt.show()


# 2. Filtriamo per il modello desiderato
df_XGB = df[df['Modello'] == 'XGB']

# 3. Creiamo la griglia vuota (un grafico per ogni cutoff)
griglia = sns.FacetGrid(df_XGB, col="cutoff", col_wrap=3, height=4)

# 4. Disegniamo i dati (k_skb va in automatico sull'asse X!)
griglia.map(plt.errorbar, "k_skb", "Media_f1_macro", "Deviazione_standard_f1_macro", marker="o", capsize=4)

# 5. AGGIUNTA: Rimettiamo le etichette per far capire a chi legge cosa stiamo guardando
griglia.set_axis_labels("Numero di feature (k_skb)", "Media F1 Macro")

griglia.figure.suptitle(
    'XGB: Media F1 Macro vs Numero di feature (k_skb) per ogni Cutoff',
    fontweight='bold',
    fontsize=14
)
griglia.figure.subplots_adjust(top=0.9)
plt.savefig('grafico_xgb_k_skb.png')  # Salva il grafico come immagine
plt.show()


# 3. Creiamo la griglia vuota (un grafico per ogni cutoff)
griglia = sns.FacetGrid(df_XGB, col="k_skb", col_wrap=5, height=4)

# 4. Disegniamo i dati (k_skb va in automatico sull'asse X!)
griglia.map(plt.errorbar, "cutoff", "Media_f1_macro", "Deviazione_standard_f1_macro", marker="o", capsize=4)

# 5. AGGIUNTA: Rimettiamo le etichette per far capire a chi legge cosa stiamo guardando
griglia.set_axis_labels("Cutoff", "Media F1 Macro")
griglia.figure.suptitle(
    'XGB: Media F1 Macro vs Cutoff per ogni Numero di feature (k_skb)',
    fontweight='bold',
    fontsize=14
)
griglia.figure.subplots_adjust(top=0.9)
plt.savefig('grafico_xgb_cutoff.png')  # Salva il grafico come immagine
plt.show()


# 2. Filtriamo per il modello desiderato
df_SVM = df[df['Modello'] == 'SVM']

# 3. Creiamo la griglia vuota (un grafico per ogni cutoff)
griglia = sns.FacetGrid(df_SVM, col="cutoff", col_wrap=3, height=4)

# 4. Disegniamo i dati (k_skb va in automatico sull'asse X!)
griglia.map(plt.errorbar, "k_skb", "Media_f1_macro", "Deviazione_standard_f1_macro", marker="o", capsize=4)

# 5. AGGIUNTA: Rimettiamo le etichette per far capire a chi legge cosa stiamo guardando
griglia.set_axis_labels("Numero di feature (k_skb)", "Media F1 Macro")
griglia.figure.suptitle(
    'SVM: Media F1 Macro vs Numero di feature (k_skb) per ogni Cutoff',
    fontweight='bold',
    fontsize=14
)
griglia.figure.subplots_adjust(top=0.9)
plt.savefig('grafico_svm_k_skb.png')  # Salva il grafico come immagine
plt.show()


# 3. Creiamo la griglia vuota (un grafico per ogni cutoff)
griglia = sns.FacetGrid(df_SVM, col="k_skb", col_wrap=5, height=4)

# 4. Disegniamo i dati (k_skb va in automatico sull'asse X!)
griglia.map(plt.errorbar, "cutoff", "Media_f1_macro", "Deviazione_standard_f1_macro", marker="o", capsize=4)

# 5. AGGIUNTA: Rimettiamo le etichette per far capire a chi legge cosa stiamo guardando
griglia.set_axis_labels("Cutoff", "Media F1 Macro")
griglia.figure.suptitle(
    'SVM: Media F1 Macro vs Cutoff per ogni Numero di feature (k_skb)',
    fontweight='bold',
    fontsize=14
)
griglia.figure.subplots_adjust(top=0.9)
plt.savefig('grafico_svm_cutoff.png')  # Salva il grafico come immagine
plt.show()



### Prova solo DS



# 2. Filtriamo per il modello desiderato
df_RF = df[df['Modello'] == 'Random Forest']

# 3. Creiamo la griglia vuota (un grafico per ogni cutoff)
griglia = sns.FacetGrid(df_RF, col="cutoff", col_wrap=3, height=4)

# 4. Disegniamo i dati (k_skb va in automatico sull'asse X!)
griglia.map(plt.plot, "k_skb", "Deviazione_standard_f1_macro", marker="o")

# 5. AGGIUNTA: Rimettiamo le etichette per far capire a chi legge cosa stiamo guardando
griglia.set_axis_labels("Numero di feature (k_skb)", "Deviazione Standard F1 Macro")
griglia.figure.suptitle(
    'Random Forest: Deviazione Standard F1 Macro vs Numero di feature (k_skb) per ogni Cutoff',
    fontweight='bold',
    fontsize=14
)
griglia.figure.subplots_adjust(top=0.9)
plt.savefig('grafico_rf_deviazione_k_skb.png')  # Salva il grafico come immagine
plt.show()


# 3. Creiamo la griglia vuota (un grafico per ogni cutoff)
griglia = sns.FacetGrid(df_RF, col="k_skb", col_wrap=5, height=4)

# 4. Disegniamo i dati (k_skb va in automatico sull'asse X!)
griglia.map(plt.plot, "cutoff", "Deviazione_standard_f1_macro", marker="o")

# 5. AGGIUNTA: Rimettiamo le etichette per far capire a chi legge cosa stiamo guardando
griglia.set_axis_labels("Cutoff", "Deviazione Standard F1 Macro")
griglia.figure.suptitle(
    'Random Forest: Deviazione Standard F1 Macro vs Cutoff per ogni Numero di feature (k_skb)',
    fontweight='bold',
    fontsize=14
)
griglia.figure.subplots_adjust(top=0.9)
plt.savefig('grafico_rf_deviazione_cutoff.png')  # Salva il grafico come immagine
plt.show()



# 3. Creiamo la griglia vuota (un grafico per ogni cutoff)
griglia = sns.FacetGrid(df_XGB, col="cutoff", col_wrap=3, height=4)

# 4. Disegniamo i dati (k_skb va in automatico sull'asse X!)
griglia.map(plt.plot, "k_skb", "Deviazione_standard_f1_macro", marker="o")

# 5. AGGIUNTA: Rimettiamo le etichette per far capire a chi legge cosa stiamo guardando
griglia.set_axis_labels("Numero di feature (k_skb)", "Deviazione Standard F1 Macro")
griglia.figure.suptitle(
    'XGB: Deviazione Standard F1 Macro vs Numero di feature (k_skb) per ogni Cutoff',
    fontweight='bold',
    fontsize=14
)
griglia.figure.subplots_adjust(top=0.9)
plt.savefig('grafico_xgb_deviazione_k_skb.png')  # Salva il grafico come immagine
plt.show()


# 3. Creiamo la griglia vuota (un grafico per ogni cutoff)
griglia = sns.FacetGrid(df_XGB, col="k_skb", col_wrap=5, height=4)

# 4. Disegniamo i dati (k_skb va in automatico sull'asse X!)
griglia.map(plt.plot, "cutoff", "Deviazione_standard_f1_macro", marker="o")

# 5. AGGIUNTA: Rimettiamo le etichette per far capire a chi legge cosa stiamo guardando
griglia.set_axis_labels("Cutoff", "Deviazione Standard F1 Macro")
griglia.figure.suptitle(
    'XGB: Deviazione Standard F1 Macro vs Cutoff per ogni Numero di feature (k_skb)',
    fontweight='bold',
    fontsize=14
)
griglia.figure.subplots_adjust(top=0.9)
plt.savefig('grafico_xgb_deviazione_cutoff.png')  # Salva il grafico come immagine
plt.show()




# 3. Creiamo la griglia vuota (un grafico per ogni cutoff)
griglia = sns.FacetGrid(df_SVM, col="cutoff", col_wrap=3, height=4)

# 4. Disegniamo i dati (k_skb va in automatico sull'asse X!)
griglia.map(plt.plot, "k_skb", "Deviazione_standard_f1_macro", marker="o")

# 5. AGGIUNTA: Rimettiamo le etichette per far capire a chi legge cosa stiamo guardando
griglia.set_axis_labels("Numero di feature (k_skb)", "Deviazione Standard F1 Macro")
griglia.figure.suptitle(
    'SVM: Deviazione Standard F1 Macro vs Numero di feature (k_skb) per ogni Cutoff',
    fontweight='bold',
    fontsize=14
)
griglia.figure.subplots_adjust(top=0.9)
plt.savefig('grafico_svm_deviazione_k_skb.png')  # Salva il grafico come immagine
plt.show()


# 3. Creiamo la griglia vuota (un grafico per ogni cutoff)
griglia = sns.FacetGrid(df_SVM, col="k_skb", col_wrap=5, height=4)

# 4. Disegniamo i dati (k_skb va in automatico sull'asse X!)
griglia.map(plt.plot, "cutoff", "Deviazione_standard_f1_macro", marker="o")

# 5. AGGIUNTA: Rimettiamo le etichette per far capire a chi legge cosa stiamo guardando
griglia.set_axis_labels("Cutoff", "Deviazione Standard F1 Macro")
griglia.figure.suptitle(
    'SVM: Deviazione Standard F1 Macro vs Cutoff per ogni Numero di feature (k_skb)',
    fontweight='bold',
    fontsize=14
)
griglia.figure.subplots_adjust(top=0.9)
plt.savefig('grafico_svm_deviazione_cutoff.png')  # Salva il grafico come immagine
plt.show()




# GRAFICI COMPARATIVI PER TUTTI E TRE I MODELLI


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Carichiamo la tabella
df = pd.read_csv('risultati_crossvalidation_skb.csv')

# Filtriamo per assicurarci di avere solo i tre modelli desiderati nel grafico
df_modelli = df[df['Modello'].isin(['Random Forest', 'XGB', 'SVM'])]

# ==========================================
# PARTE 1: GRAFICI PER MEDIA F1 MACRO
# ==========================================

# --- 1A: Media F1 Macro vs Numero di feature (k_skb) ---
g1 = sns.catplot(
    data=df_modelli, 
    kind="point", 
    x="k_skb", 
    y="Media_f1_macro", 
    hue="Modello", 
    col="cutoff", 
    col_wrap=3, 
    height=4, 
    markers=["o", "s", "D"],
    dodge=True # Sfasa leggermente i punti per non sovrapporli
)

g1.set_axis_labels("Numero di feature (k_skb)", "Media F1 Macro")
g1.figure.suptitle(
    'Confronto Modelli: Media F1 Macro vs Numero di feature (k_skb) per ogni Cutoff', 
    fontweight='bold', 
    fontsize=14
)
g1.figure.subplots_adjust(top=0.9)
plt.savefig('confronto_modelli_media_k_skb.png', bbox_inches='tight')
plt.show()


# --- 1B: Media F1 Macro vs Cutoff ---
g2 = sns.catplot(
    data=df_modelli, 
    kind="point", 
    x="cutoff", 
    y="Media_f1_macro", 
    hue="Modello", 
    col="k_skb", 
    col_wrap=5, 
    height=4, 
    markers=["o", "s", "D"],
    dodge=True
)

g2.set_axis_labels("Cutoff", "Media F1 Macro")
g2.figure.suptitle(
    'Confronto Modelli: Media F1 Macro vs Cutoff per ogni Numero di feature (k_skb)', 
    fontweight='bold', 
    fontsize=14
)
g2.figure.subplots_adjust(top=0.9)
plt.savefig('confronto_modelli_media_cutoff.png', bbox_inches='tight')
plt.show()


# ==========================================
# PARTE 2: GRAFICI PER DEVIAZIONE STANDARD
# ==========================================

# --- 2A: Deviazione Standard vs Numero di feature (k_skb) ---
g3 = sns.catplot(
    data=df_modelli, 
    kind="point", 
    x="k_skb", 
    y="Deviazione_standard_f1_macro", 
    hue="Modello", 
    col="cutoff", 
    col_wrap=3, 
    height=4, 
    markers=["o", "s", "D"],
    dodge=True
)

g3.set_axis_labels("Numero di feature (k_skb)", "Deviazione Standard F1 Macro")
g3.figure.suptitle(
    'Confronto Modelli: Deviazione Standard vs Numero di feature (k_skb) per ogni Cutoff', 
    fontweight='bold', 
    fontsize=14
)
g3.figure.subplots_adjust(top=0.9)
plt.savefig('confronto_modelli_ds_k_skb.png', bbox_inches='tight')
plt.show()


# --- 2B: Deviazione Standard vs Cutoff ---
g4 = sns.catplot(
    data=df_modelli, 
    kind="point", 
    x="cutoff", 
    y="Deviazione_standard_f1_macro", 
    hue="Modello", 
    col="k_skb", 
    col_wrap=5, 
    height=4, 
    markers=["o", "s", "D"],
    dodge=True
)

g4.set_axis_labels("Cutoff", "Deviazione Standard F1 Macro")
g4.figure.suptitle(
    'Confronto Modelli: Deviazione Standard vs Cutoff per ogni Numero di feature (k_skb)', 
    fontweight='bold', 
    fontsize=14
)
g4.figure.subplots_adjust(top=0.9)
plt.savefig('confronto_modelli_ds_cutoff.png', bbox_inches='tight')
plt.show()



# HEATMAP
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Carichiamo la tabella
df = pd.read_csv('risultati_crossvalidation_skb.csv')
df_modelli = df[df['Modello'].isin(['Random Forest', 'XGB', 'SVM'])]

# Definiamo i modelli da ciclare
modelli = ['Random Forest', 'XGB', 'SVM']

# ==========================================
# HEATMAP PER LA MEDIA F1 MACRO
# ==========================================

# Creiamo una figura con 3 grafici (subplot) affiancati
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

for i, modello in enumerate(modelli):
    # Filtriamo i dati per il singolo modello
    df_temp = df_modelli[df_modelli['Modello'] == modello]
    
    # Creiamo la matrice: Asse Y = cutoff, Asse X = k_skb, Valori = Media
    matrice = df_temp.pivot_table(index='cutoff', columns='k_skb', values='Media_f1_macro')
    
    # Invertiamo l'asse Y in modo che i cutoff più alti siano in alto (opzionale ma consigliato)
    matrice = matrice.sort_index(ascending=False)
    
    # Creiamo la heatmap sull'asse specifico (axes[i])
    sns.heatmap(
        matrice, 
        ax=axes[i], 
        annot=True,       # Scrive il valore numerico dentro ogni cella
        fmt=".3f",        # Formato a 3 cifre decimali (es. 0.852)
        cmap="YlGnBu",    # Palette di colori (dal giallo=basso al blu scuro=alto)
        cbar=(i == 2),    # Mostra la barra dei colori solo nell'ultimo grafico a destra
        linewidths=.5     # Aggiunge una griglia leggera tra le celle
    )
    
    # Titolo del singolo subplot
    axes[i].set_title(modello, fontweight='bold', fontsize=12)
    axes[i].set_ylabel('Cutoff')
    axes[i].set_xlabel('Numero di feature (k_skb)')

# Titolo globale in grassetto
fig.suptitle('Mappe di Calore: Media F1 Macro (Cutoff vs k_skb)', fontweight='bold', fontsize=16)

# Ottimizza gli spazi per non tagliare etichette o titoli
plt.tight_layout()
fig.subplots_adjust(top=0.88)

# Salvataggio e visualizzazione
plt.savefig('heatmap_modelli_media.png', dpi=300, bbox_inches='tight')
plt.show()


# ==========================================
# HEATMAP PER LA DEVIAZIONE STANDARD
# ==========================================
# (Stessa identica logica, ma per valutare la stabilità)

fig_ds, axes_ds = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

for i, modello in enumerate(modelli):
    df_temp = df_modelli[df_modelli['Modello'] == modello]
    matrice_ds = df_temp.pivot_table(index='cutoff', columns='k_skb', values='Deviazione_standard_f1_macro')
    matrice_ds = matrice_ds.sort_index(ascending=False)
    
    sns.heatmap(
        matrice_ds, 
        ax=axes_ds[i], 
        annot=True, 
        fmt=".3f", 
        cmap="OrRd",     # Palette dal bianco al rosso scuro (il rosso indica alta deviazione = peggio)
        cbar=(i == 2),
        linewidths=.5
    )
    
    axes_ds[i].set_title(modello, fontweight='bold', fontsize=12)
    axes_ds[i].set_ylabel('Cutoff')
    axes_ds[i].set_xlabel('Numero di feature (k_skb)')

fig_ds.suptitle('Mappe di Calore: Deviazione Standard F1 Macro (Cutoff vs k_skb)', fontweight='bold', fontsize=16)

plt.tight_layout()
fig_ds.subplots_adjust(top=0.88)

plt.savefig('heatmap_modelli_ds.png', dpi=300, bbox_inches='tight')
plt.show()