import pandas as pd
df1 = pd.read_csv('Fase1_Benchmark_Riduzioni_pt1.csv')
df2 = pd.read_excel('Fase1pt2_Benchmark_Riduzioni.xlsx')

df_totale = pd.concat([df1, df2], ignore_index=True)
df_totale.to_csv('Benchmark_totale.csv', index=False)