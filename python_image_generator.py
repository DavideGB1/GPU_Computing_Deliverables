import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patheffects as path_effects
outline = [path_effects.withStroke(linewidth=3, foreground="black")]
df = pd.read_csv('stats.csv', skipinitialspace=True)
plt.figure(figsize=(12, 6))

df['matrix'] = df['matrix'].str.replace("matrices/", "", regex=False).str.replace(".mtx", "", regex=False)
df['kernel'] = df['kernel'].str.replace("SELLpack-32", "SELL-32", regex=False)

ordering = ['Queen_4147', 'kmer_U1a', 'spal_004', 'audikw_1', 'F1', 'coPapersCiteseer', 'hollywood-2009', 'Maragal_8', 'Raj1','FullChip']
df['matrix'] = pd.Categorical(df['matrix'], categories=ordering, ordered=True)

ordering_kernel = ['COO', 'CSR-Scalar', 'CSR-Vector', 'CSR-Vector Shuffle', "SELL-32", "cuSPARSE"]
df['kernel'] = pd.Categorical(df['kernel'], categories=ordering_kernel, ordered=True)

df = df.rename(columns={"matrix": "Matrix", "bandwidth_avg": "Bandwidth(GB/s)", "gflops_avg": "GFLOP/s"})


fig, ax = plt.subplots(figsize=(16, 7))
sns.barplot(x="Matrix", y="GFLOP/s",hue="kernel",edgecolor="black",capsize=.05,errorbar="sd", data=df, width=0.9)
plt.gca().legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
          ncol=6,fontsize="15")

for container in ax.containers:
    labels = [f'{v:.2f}'  for v in container.datavalues]
    texts = ax.bar_label(container, labels=labels, fontsize=11, rotation=90,
                 label_type='center', color='white',fontweight='bold')
    for t in texts:
        t.set_path_effects(outline)
#plt.text(x=-0.49,y=1000,s="933(GB/s)",color='red',fontweight='bold',fontsize=15)
#plt.axhline(y=933, color='red', linestyle='--')
#plt.yscale("log")
plt.xticks(rotation=10)
ax.set_xlabel('Matrix', fontsize = 30)
ax.set_ylabel('GFLOP/s', fontsize = 30)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.tick_params(axis='both', which='minor', labelsize=25)
plt.tight_layout()
#plt.savefig("GFLOPS.svg")
plt.show()

fig, ax = plt.subplots(figsize=(16, 7))
sns.barplot(x="Matrix", y="Bandwidth(GB/s)",hue="kernel",edgecolor="black",capsize=.05,errorbar="sd", data=df, width=0.9)
plt.gca().legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
          ncol=6,fontsize="15")

for container in ax.containers:
    labels = [f'{v:.2f}' if v > 2 else '1' for v in container.datavalues]
    texts = ax.bar_label(container, labels=labels, fontsize=11, rotation=90,
                 label_type='center', color='white',fontweight='bold')
    for t in texts:
        t.set_path_effects(outline)
plt.text(x=-0.49,y=942,s="933(GB/s A30 Theoretical Peak)",color='red',fontweight='bold',fontsize=15)
plt.axhline(y=933, color='red', linestyle='--')
#plt.yscale("symlog")
plt.xticks(rotation=10)
ax.set_xlabel('Matrix', fontsize = 30)
ax.set_ylabel('Bandwidth(GB/s)', fontsize = 30)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.tick_params(axis='both', which='minor', labelsize=25)
plt.tight_layout()
#plt.savefig("Bandwidth.svg")
plt.show()
