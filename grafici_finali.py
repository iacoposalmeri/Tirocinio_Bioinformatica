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

# 1. Mappa esatta delle feature dopo il filtro prevalenza (dai tuoi dati)
MAPPA_PREVALENZA = {
    0.03: 383,
    0.05: 318,
    0.07: 272,
    0.1: 234,
    0.15: 201,
    0.2: 171
}

# 2. Caricamento Dati
df = pd.read_csv('Gridsearch_finale.csv')
# Separiamo la colonna Technique in Trasformazione e Metodo (es. 'CLR_RFE' -> 'CLR' e 'RFE')
df[['Transformation', 'FS_Method']] = df['Technique'].str.split('_', expand=True)
# Gestione del 'Consensus' che non ha l'underscore iniziale nel tuo CSV
df.loc[df['Technique'] == 'Consensus', 'FS_Method'] = 'Consensus'
df.loc[df['Technique'] == 'Consensus', 'Transformation'] = 'Nessuna (Consensus)'

# ---------------------------------------------------------
# 3. SCEGLI LA TUA PIPELINE SPECIFICA DA PLOTTARE
# ---------------------------------------------------------
MIGLIOR_MODELLO = 'XGB'           # Es. 'RF' o 'XGB'
MIGLIORE_TRASFORMAZIONE = 'CLR'  # Es. 'CLR' o 'RCLR'
MIGLIOR_FS = 'SKB'               # Es. 'SKB', 'RFE', 'ElasticNet' o 'Consensus'
MIGLIOR_CUTOFF = 0.05        # Scegli il cutoff

# Filtriamo per estrarre l'UNICA riga corrispondente alla tua scelta
df_best = df[(df['Model'] == MIGLIOR_MODELLO) & 
             (df['Transformation'] == MIGLIORE_TRASFORMAZIONE) & 
             (df['FS_Method'] == MIGLIOR_FS) & 
             (df['Cutoff'] == MIGLIOR_CUTOFF)]

if df_best.empty:
    print("Attenzione: Combinazione non trovata nel CSV! Controlla i nomi.")
else:
    # 4. Estrazione del numero finale di feature (Biomarcatori)
    riga = df_best.iloc[0]
    params = ast.literal_eval(riga['Best_Params'])
    metodo = riga['FS_Method']
    
    if metodo == 'SKB':
        num_final_features = params.get('skb__k', 0)
    elif metodo == 'RFE':
        num_final_features = params.get('rfe__n_features_to_select', 0)
    elif metodo == 'ElasticNet':
        num_final_features = params.get('elasticnet__max_features', 0)
    elif metodo == 'Consensus':
        num_final_features = params.get('consensus__k', 0)
    else:
        num_final_features = 0

    # Ricaviamo il numero intermedio dalla nostra mappa automatica
    num_intermedio = MAPPA_PREVALENZA.get(MIGLIOR_CUTOFF, 934)

    # 5. Costruzione del Dataset per le 3 barre perfette
    dati_plot = {
        'Fase della Pipeline': [
            '1. Dati Grezzi\n(100% Taxa)', 
            f'2. Filtro Prevalenza\n(Cutoff {MIGLIOR_CUTOFF})', 
            f'3. Feature Selection\n({MIGLIOR_FS})'
        ],
        'Numero Feature': [934, num_intermedio, num_final_features],
        'Categoria': ['Baseline', 'Baseline', 'Biomarcatori']
    }
    df_plot = pd.DataFrame(dati_plot)

    # 6. Creazione del Grafico a Imbuto
    plt.figure(figsize=(10, 6))

    ax = sns.barplot(
        data=df_plot, 
        x='Fase della Pipeline', 
        y='Numero Feature',
        hue='Categoria',
        palette={'Baseline': '#bdc3c7', 'Biomarcatori': '#e74c3c'}, # Grigio per i passaggi, Rosso per il target finale
        edgecolor='black',
        dodge=False
    )

    # Aggiunge i numeri esatti sopra le barre
    for p in ax.patches:
        altezza = p.get_height()
        if altezza > 0: 
            ax.annotate(f'{int(altezza)}', 
                        (p.get_x() + p.get_width() / 2., altezza), 
                        ha='center', va='center', 
                        fontsize=13, fontweight='bold', color='black', 
                        xytext=(0, 10), textcoords='offset points')

    plt.title(f'Riduzione della Dimensionalità: Pipeline {MIGLIOR_MODELLO} + {MIGLIORE_TRASFORMAZIONE} + {MIGLIOR_FS}', 
              fontsize=15, fontweight='bold', pad=20)
    plt.ylabel('Numero di Specie Batteriche', fontsize=12)
    plt.xlabel('') # Tolto perché i nomi delle fasi sono già chiari
    plt.legend(title='', loc='upper right')

    # Opzionale: Salvataggio
    # plt.savefig(f'Imbuto_Feature_{MIGLIOR_MODELLO}_{MIGLIOR_FS}.png', dpi=300, bbox_inches='tight')

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