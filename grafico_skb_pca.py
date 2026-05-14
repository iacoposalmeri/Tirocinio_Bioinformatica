import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. CARICAMENTO DATI E SETUP
# ==========================================
df = pd.read_csv('risultati_crossvalidation_skb_pca.csv')

# Estrapoliamo i modelli e i valori di K (SKB)
modelli = df['Modello'].unique()
k_values = sorted(df['k_skb'].unique())

# Impostiamo lo stile grafico
sns.set_theme(style="whitegrid")

# Trova i valori min e max globali per avere una scala uniforme
vmin_val = df['Media_f1_macro'].min()
vmax_val = df['Media_f1_macro'].max()

# ==========================================
# 2. GENERAZIONE HEATMAP (F1 + N. COMPONENTI)
# ==========================================
for modello in modelli:
    df_modello = df[df['Modello'] == modello]
    
    # Creiamo un pannello con un sotto-grafico per ogni K
    fig, axes = plt.subplots(1, len(k_values), figsize=(20, 6))
    fig.suptitle(f'Heatmap F1-Macro: {modello} (SKB + PCA)', fontweight='bold', fontsize=16)
    
    # Gestione del caso in cui ci sia un solo valore di K
    if len(k_values) == 1:
        axes = [axes]
    
    for i, k in enumerate(k_values):
        df_k = df_modello[df_modello['k_skb'] == k]
        
        if df_k.empty: 
            continue
        
        # 1. Matrice per i valori dell'F1-Macro (determina il colore)
        matrice_f1 = df_k.pivot_table(index='cutoff', columns='varianza_pca', values='Media_f1_macro')
        matrice_f1 = matrice_f1.sort_index(ascending=False)
        
        # 2. Matrice per i valori del numero di componenti (solo per il testo)
        matrice_n = df_k.pivot_table(index='cutoff', columns='varianza_pca', values='n_componenti')
        matrice_n = matrice_n.sort_index(ascending=False)
        
        # 3. Creazione dinamica del testo dentro le celle (F1 + N. Componenti)
        annot_labels = np.empty_like(matrice_f1, dtype=object)
        for r in range(matrice_f1.shape[0]):
            for c in range(matrice_f1.shape[1]):
                valore_f1 = matrice_f1.iloc[r, c]
                valore_n = matrice_n.iloc[r, c]
                # Se la cella non è vuota, scrivi il testo composto
                if pd.notna(valore_f1):
                    annot_labels[r, c] = f"{valore_f1:.3f}\n(n={int(valore_n)})"
                else:
                    annot_labels[r, c] = ""

        # Disegniamo la heatmap
        sns.heatmap(
            matrice_f1, 
            ax=axes[i], 
            annot=annot_labels, 
            fmt="",  # Lasciamo vuoto perché abbiamo già formattato il testo personalizzato!
            cmap="YlGnBu", 
            cbar=(i == len(k_values)-1), 
            vmin=vmin_val, 
            vmax=vmax_val,
            linewidths=.5
        )
        
        axes[i].set_title(f'Feature Iniziali (SKB): {k}', fontweight='bold')
        axes[i].set_ylabel('Cutoff Prevalenza' if i == 0 else '')
        axes[i].set_xlabel('Varianza PCA')

    plt.tight_layout()
    fig.subplots_adjust(top=0.85)
    
    # Salvataggio
    nome_modello_pulito = modello.replace(" ", "_")
    plt.savefig(f'Heatmap_Completa_{nome_modello_pulito}.png', dpi=300, bbox_inches='tight')
    plt.close()

# ==========================================
# 3. GRAFICI DI TREND (LINEE/SCATTER)
# ==========================================

# Trend 1: F1-Macro vs Numero Feature Iniziali (SKB)
plt.figure(figsize=(10, 6))
sns.pointplot(data=df, x='k_skb', y='Media_f1_macro', hue='Modello', markers=['o', 's', 'D'], dodge=True)
plt.title('Performance Media per Numero di Feature Iniziali (SKB)', fontweight='bold', fontsize=14)
plt.xlabel('Numero di Feature Selezionate (k_skb)')
plt.ylabel('Media F1-Macro')
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('Trend_K_SKB.png', dpi=300, bbox_inches='tight')
plt.close()

# Trend 2: F1-Macro vs Varianza PCA
plt.figure(figsize=(10, 6))
sns.pointplot(data=df, x='varianza_pca', y='Media_f1_macro', hue='Modello', markers=['o', 's', 'D'], dodge=True)
plt.title('Performance Media in base alla Varianza Spiegata (PCA)', fontweight='bold', fontsize=14)
plt.xlabel('Varianza Spiegata (PCA)')
plt.ylabel('Media F1-Macro')
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('Trend_Varianza_PCA.png', dpi=300, bbox_inches='tight')
plt.close()

print("Generazione completata! Heatmap doppie e trend salvati con successo.")