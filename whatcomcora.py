
import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np

mu = np.array([1, 3, 5, 7, 9])

data = {
    'Cora':    np.array([92.65, 93.34, 93.12, 92.48, 92.17]),
    'Citeseer':np.array([94.56, 94.78, 95.41, 94.92, 94.39]),
    'Pubmed':  np.array([82.01, 83.64, 84.11, 83.40, 81.83]),
}

mu_scaled = {
    'Cora': mu * 8,
    'Citeseer': mu * 8,
    'Pubmed': mu * 8,
}

plt.figure(figsize=(8,5))

for dataset in data:
    plt.plot(mu_scaled[dataset], data[dataset], marker='o', label=dataset)

all_ticks = np.unique(np.concatenate([mu_scaled[d] for d in mu_scaled]))
plt.xticks(all_ticks)
plt.gca().set_xticklabels([f'{int(t)}' for t in all_ticks])
plt.tick_params(axis='x', direction='in', length=6)

# 去掉竖着的虚线（网格）
plt.grid(False)



plt.xlabel('K')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()
