# Analisi del Microbioma: PCA Tradizionale vs Aitchison PCA

Questa repository contiene la pipeline sviluppata per affrontare il problema di composizionalità nei dati del microbioma intestinale e prevedere lo stato di salute dei pazienti nel caso CRC vs Control (o Healthy, con opportune modifiche)

## Struttura dei File
* **`utils_crc.py`**: Modulo contenente le funzioni per il pre-processing dei dati
* **`main_PCA_Tradizionale.py`**: Analisi esplorativa con PCA standard sui dati grezzi, per evidenziare i limiti geometrici e il bias composizionale.
* **`main_PCA_Aitchison.py`**: Applica la trasformazione CLR, esegue la PCA e addestra il modello predittivo (Random Forest) validato tramite Matrice di Confusione, ROC e MCC.
* **`cutoff_experiments.py`**: Script contenente i test iterativi per vedere come le riduzioni di dimensionalità con diversi cutoff influiscono sulla qualità di diversi classificatori.