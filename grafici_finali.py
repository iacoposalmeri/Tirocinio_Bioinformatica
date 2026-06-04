import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from utils_crc2 import * 
###HEATMAP###

# 1. Carichiamo i dati "puliti"
df = pd.read_csv('Gridsearch_finale.csv')

# 2. Dividiamo la colonna Technique in un colpo solo (grazie alla tua modifica!)
df[['Transformation', 'FS_Method']] = df['Technique'].str.split('_', expand=True)

# 3. Prepariamo la "tela" per i nostri 4 grafici
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Performance (Test MCC) nel classificare il Cancro al Colon Retto', fontsize=18)

models = ['RF', 'XGB']
transformations = ['CLR', 'RCLR']

# 4. Creiamo le 4 Heatmap tramite il famoso Pivot
for i, model in enumerate(models):
    for j, transf in enumerate(transformations):
        ax = axes[i, j] # Seleziona il quadratino giusto nella griglia 2x2
        
        # Filtriamo le "pratiche" (come l'esempio di prima)
        subset = df[(df['Model'] == model) & (df['Transformation'] == transf)]
        
        # Facciamo il PIVOT: Cutoff sulle righe, Tecniche sulle colonne, MCC dentro le celle
        pivot_df = subset.pivot(index='Cutoff', columns='FS_Method', values='Test_MCC')
        
        # Ordiniamo in modo che il filtro più restrittivo (0.20) sia in alto, o viceversa
        pivot_df = pivot_df.sort_index(ascending=False)
        
        # Disegniamo la heatmap! Fissiamo vmin=0.4 e vmax=0.6 per avere la stessa scala colori
        sns.heatmap(pivot_df, annot=True, fmt=".3f", cmap='viridis', 
                    vmin=0.4, vmax=0.6, ax=ax, cbar_kws={'label': 'MCC Score'})
        
        ax.set_title(f'{model} + {transf}', fontsize=14, fontweight='bold')
        ax.set_ylabel('Cutoff' if j == 0 else '')
        ax.set_xlabel('Tecnica' if i == 1 else '')

plt.tight_layout()
plt.show()


###BARPLOT TOP20###
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Caricamento e preparazione iniziale (identico a prima)
x, y_binary, _, _ = caricamento_pulizia_dati("Metadati_CRC_Dataset.csv", "Abbondanze_CRC_Dataset.csv", condizione_negativa='healthy')

df_plot = x.copy()
df_plot['Classe'] = y_binary.map({0: 'Sani', 1: 'CRC'})
# 1. Calcoliamo le medie e stiliamo le due classifiche (come prima)
medie_sani = df_plot[df_plot['Classe'] == 'Sani'].drop('Classe', axis=1).mean()
medie_crc = df_plot[df_plot['Classe'] == 'CRC'].drop('Classe', axis=1).mean()

top20_sani = medie_sani.sort_values(ascending=False).head(20).index.tolist()
top20_crc = medie_crc.sort_values(ascending=False).head(20).index.tolist()

# 2. Prepariamo i due dataset
df_sani_top = df_plot[['Classe'] + top20_sani].melt(id_vars=['Classe'], var_name='Specie', value_name='Abbondanza')
df_crc_top = df_plot[['Classe'] + top20_crc].melt(id_vars=['Classe'], var_name='Specie', value_name='Abbondanza')

# 3. Creazione della Figura con 2 grafici (uno sopra l'altro)
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 18))

# --- GRAFICO IN ALTO: TOP 20 SANI (Confronto Sani vs CRC) ---
sns.barplot(
    data=df_sani_top, 
    x='Abbondanza', 
    y='Specie', 
    hue='Classe',
    palette={'Sani': '#2ecc71', 'CRC': '#e74c3c'}, 
    order=top20_sani,
    errorbar=('ci', 95), # Aggiunge le barre di errore (Intervallo di confidenza al 95%)
    capsize=0.15,        # Aggiunge i "trattini" alle estremità delle barre di errore
    ax=axes[0]
)
# A differenza del boxplot, qui NON usiamo la scala logaritmica per mantenere l'intuizione visiva lineare
axes[0].set_title('Abbondanze Medie: Top 20 Specie dominanti nei SANI', fontsize=16, fontweight='bold')
axes[0].set_xlabel('Abbondanza Relativa Media', fontsize=12)
axes[0].set_ylabel('')

# --- GRAFICO IN BASSO: TOP 20 CRC (Confronto Sani vs CRC) ---
sns.barplot(
    data=df_crc_top, 
    x='Abbondanza', 
    y='Specie', 
    hue='Classe',
    palette={'Sani': '#2ecc71', 'CRC': '#e74c3c'}, 
    order=top20_crc,
    errorbar=('ci', 95),
    capsize=0.15,
    ax=axes[1]
)
axes[1].set_title('Abbondanze Medie: Top 20 Specie dominanti nel Cancro al Colon-Retto (CRC)', fontsize=16, fontweight='bold')
axes[1].set_xlabel('Abbondanza Relativa Media', fontsize=12)
axes[1].set_ylabel('')

# Estetica finale
plt.tight_layout()
plt.show()





import pandas as pd
import ast
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Carica i dati
df = pd.read_csv('Gridsearch_finale.csv')
df[['Transformation', 'FS_Method']] = df['Technique'].str.split('_', expand=True)

# ---------------------------------------------------------
# 2. SCEGLI LA COMBINAZIONE ESATTA CHE VUOI GRAFICARE!
# ---------------------------------------------------------
MIGLIOR_MODELLO = 'XGB'         # Scegli tra 'RF' o 'XGB'
MIGLIORE_TRASFORMAZIONE = 'CLR' # Scegli tra 'CLR' o 'RCLR'
MIGLIOR_CUTOFF = 0.05           # Scegli il cutoff (es. 0.03, 0.07, ecc.)

# Filtriamo il dataset usando le tue scelte
df_best = df[(df['Model'] == MIGLIOR_MODELLO) & 
             (df['Transformation'] == MIGLIORE_TRASFORMAZIONE) & 
             (df['Cutoff'] == MIGLIOR_CUTOFF)].copy()

# 3. Funzione per estrarre il numero di feature (rimasta identica)
def estrai_num_feature(riga):
    params = ast.literal_eval(riga['Best_Params'])
    metodo = riga['FS_Method']
    
    if metodo == 'SKB':
        return params.get('skb__k', 0)
    elif metodo == 'RFE':
        return params.get('rfe__n_features_to_select', 0)
    elif metodo == 'ElasticNet':
        return params.get('elasticnet__max_features', 0)
    elif metodo == 'Consensus':
        return params.get('consensus__k', 0)
    return 0

df_best['Num_Features'] = df_best.apply(estrai_num_feature, axis=1)

# Assicuriamoci che i dati siano ordinati per metodo per pulizia visiva
df_best = df_best.sort_values(by='FS_Method')

# 4. Creiamo i dati per il grafico
# Ho inserito 383 come numero di feature del Cutoff 0.03 (letto dal tuo file differenze_cutoff.csv)
dati_plot = {
    'Fase / Tecnica': ['1. Dati Grezzi', '2. Filtro Prevalenza'] + df_best['FS_Method'].tolist(),
    'Numero Feature': [934, 383] + df_best['Num_Features'].tolist(),
    'Categoria': ['Baseline', 'Baseline'] + ['Feature Selection'] * len(df_best)
}
df_plot = pd.DataFrame(dati_plot)

# 5. Creazione del Grafico
plt.figure(figsize=(12, 7))

ax = sns.barplot(
    data=df_plot, 
    x='Fase / Tecnica', 
    y='Numero Feature',
    hue='Categoria',
    palette={'Baseline': '#bdc3c7', 'Feature Selection': '#3498db'},
    edgecolor='black',
    dodge=False
)

# Aggiunge i numeri sopra le barre
for p in ax.patches:
    if p.get_height() > 0: # Evita di mettere lo 0 se la barra è vuota
        ax.annotate(f'{int(p.get_height())}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', 
                    fontsize=12, fontweight='bold', color='black', 
                    xytext=(0, 10), textcoords='offset points')

plt.title(f'Feature Selection ({MIGLIOR_MODELLO} + {MIGLIORE_TRASFORMAZIONE} | Cutoff: {MIGLIOR_CUTOFF})', 
          fontsize=16, fontweight='bold', pad=15)
plt.ylabel('Numero di Specie Batteriche', fontsize=12)
plt.xlabel('Fase della Pipeline / Algoritmo', fontsize=12)
plt.legend(title='', loc='upper right')

# plt.savefig(f'riduzione_feature_{MIGLIOR_MODELLO}_{MIGLIORE_TRASFORMAZIONE}_{MIGLIOR_CUTOFF}.png', dpi=300)

plt.tight_layout()
plt.show()





import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Carichiamo i dati dal tuo file
df_cutoff = pd.read_csv('differenze_cutoff.csv')

# 2. Riordiniamo i dati in modo logico
# Vogliamo "none" all'inizio (come baseline) e poi i cutoff dal più basso al più alto
mappa_ordine = {'none': 0, '0.03': 1, '0.05': 2, '0.07': 3, '0.1': 4, '0.15': 5, '0.2': 6}

# Assicuriamoci che la colonna cutoff sia letta come stringa per il mapping
df_cutoff['cutoff_str'] = df_cutoff['cutoff'].astype(str)
df_cutoff['ordine'] = df_cutoff['cutoff_str'].map(mappa_ordine)
df_cutoff = df_cutoff.sort_values('ordine')

# 3. Creazione del Grafico
plt.figure(figsize=(10, 6))

# Usiamo una palette sequenziale che diventa più scura/intensa man mano che il filtro stringe
ax = sns.barplot(
    data=df_cutoff,
    x='cutoff_str',
    y='feature_rimaste',
    palette='Blues_d', # Ottima per le tesi/pubblicazioni
    edgecolor='black'
)

# 4. Aggiungiamo il numero esatto sopra ogni barra
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', 
                fontsize=11, fontweight='bold', color='black', 
                xytext=(0, 8), textcoords='offset points')

# 5. Estetica
plt.title('Impatto del Filtro di Prevalenza sulla Dimensionalità', fontsize=16, fontweight='bold', pad=15)
plt.ylabel('Numero di Specie Batteriche Rimaste', fontsize=12)
plt.xlabel('Soglia del Filtro di Prevalenza (Cutoff)', fontsize=12)

# Rinominare l'etichetta 'none' in qualcosa di più leggibile per chi legge
etichette_x = ['Nessun Filtro\n(Baseline)' if x == 'none' else f'{x}' for x in df_cutoff['cutoff_str']]
ax.set_xticklabels(etichette_x)

# Salvataggio del grafico (ricorda di attivarlo e sistemare il percorso!)
# plt.savefig('/percorso/del/tuo/computer/impatto_prevalenza.png', dpi=300, bbox_inches='tight')

plt.tight_layout()
plt.show()