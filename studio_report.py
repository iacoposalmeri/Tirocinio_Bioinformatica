import pandas as pd

df = pd.read_csv("/home/giosc/Scrivania/CNR/crc/Tirocinio_Bioinformatica/Fase1_Benchmark_Riduzioni.csv")
df.sort_values(by="MCC_mean", ascending=False, inplace=True)
df.head(10).to_csv("top10_benchmark_Riduzioni.csv", index=False)