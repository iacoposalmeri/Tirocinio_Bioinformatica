import pandas as pd

# Carico il dataset
df = pd.read_csv('species_yachida.tsv', sep='\t')

df.to_csv('species_yachida.csv', index=False)