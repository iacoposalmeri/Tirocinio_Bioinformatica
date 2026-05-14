import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
report_skb_pca = pd.read_csv("risultati_crossvalidation_skb_pca.csv")
report_pca = pd.read_csv("risultati_crossvalidation_pca.csv")
report_skb = pd.read_csv("risultati_crossvalidation_skb.csv")
report_skb_pca.sort_values(by="Media_f1_macro", ascending=False, inplace=True)
report_pca.sort_values(by="Media_f1_macro", ascending=False, inplace=True)
report_skb.sort_values(by="Media_f1_macro", ascending=False, inplace=True)
