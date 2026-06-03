import pandas as pd
df1 = pd.read_csv('GridSearch_FeatureSelection_clr.csv')
df2 = pd.read_csv('GridSearch_FeatureSelection.csv')

df1['Technique'] = df1['Technique'].replace({'Consensus': 'CLR_Consensus'})
df2['Technique'] = df2['Technique'].replace({'Consensus': 'RCLR_Consensus'})

df_totale = pd.concat([df1, df2], ignore_index=True)
df_totale.to_csv('Gridsearch_finale.csv', index=False)