import pandas as pd
df1 = pd.read_csv('/home/giosc/Scrivania/CNR/crc/Tirocinio_Bioinformatica/report_giorgio/GridSearch_FeatureSelection.csv')
df2 = pd.read_csv('/home/giosc/Scrivania/CNR/crc/Tirocinio_Bioinformatica/report_iacopo/GridSearch_FeatureSelection.csv')

df_totale = pd.concat([df1, df2], ignore_index=True)
df_totale.to_csv('Gridsearch_finale.csv', index=False)
