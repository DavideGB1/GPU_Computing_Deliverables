import pandas as pd
import matplotlib.pyplot as plt

# 1. Caricamento dati
# Assicurati che il file si chiami 'data.csv' nella stessa cartella
df = pd.read_csv('stats.csv', skipinitialspace=True)

# 2. Pulizia (rimuove i test falliti con errore >= 0.5)
df = df[df['avg_err'] < 0.5]

# 3. Aggregazione per block_size (media dei risultati)
summary = df.groupby('block_size')[['avg_ms', 'gflops', 'bandwidth_gbs']].mean().sort_index()
summary.index = summary.index.astype(str) # Per etichette asse X chiare

# 4. Creazione Grafico
fig, ax1 = plt.subplots(figsize=(10, 6))

# Primo asse (Y sinistra): GFLOPS e Tempo
ax1.set_xlabel('Block Size')
ax1.set_ylabel('GFLOPS / Tempo (ms)')
ax1.plot(summary.index, summary['gflops'], marker='o', color='tab:blue', label='GFLOPS', linewidth=2)
ax1.plot(summary.index, summary['avg_ms'], marker='x', color='tab:green', label='Tempo (ms)', linewidth=1)
ax1.grid(True, linestyle='--', alpha=0.6)

# Secondo asse (Y destra): Bandwidth
ax2 = ax1.twinx()
ax2.set_ylabel('Bandwidth (GB/s)', color='tab:red')
ax2.plot(summary.index, summary['bandwidth_gbs'], marker='s', color='tab:red', linestyle='--', label='Bandwidth')
ax2.tick_params(axis='y', labelcolor='tab:red')

# Titolo e Legenda
plt.title('Andamento Performance vs Block Size')
fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.85))

plt.tight_layout()
plt.show()