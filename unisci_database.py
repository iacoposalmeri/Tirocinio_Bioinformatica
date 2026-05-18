import pandas as pd
df1 = pd.read_csv('Risultati_Esplorazione_Totale.csv')
df2 = pd.read_csv('Risultati_Esplorazione_AITCHISON_PCA_HIGH_VARIANCE.csv')
df1.loc[df1['Technique'] == 'Aitchison_PCA', 'Technique'] = 'Aitchison_PCA (15 Comp)'
df2['Technique'] = 'Aitchison_PCA (90% Var)'

df_totale = pd.concat([df1, df2], ignore_index=True)
df_totale.to_csv('Risultati_Uniti.csv', index=False)