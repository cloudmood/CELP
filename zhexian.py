import matplotlib as mpl
mpl.use('tkagg')

import matplotlib.pyplot as plt
import numpy as np

mu = np.array([1, 3, 5, 7, 9])

data = {
    'Cora':     np.array([91.80, 91.86, 93.34, 92.18, 91.97]),
    'Citeseer': np.array([93.25, 93.56, 95.41, 93.27, 92.49]),
    'Pubmed':   np.array([80.01, 82.64, 84.11, 82.40, 80.31]),
    'Photo':    np.array([57.20, 57.64, 58.13, 58.87, 58.66]),
    'Computers':np.array([43.35, 43.30, 43.64, 43.71, 43.18]),
    'Collab':   np.array([66.02, 66.16, 66.87, 66.59, 66.27]),
}

plt.figure(figsize=(8,5))

for name, acc in data.items():
    plt.plot(mu, acc, marker='o', label=name)

plt.xticks(mu)
plt.gca().set_xticklabels([str(m) for m in mu])
plt.tick_params(axis='x', direction='in', length=6)

plt.grid(False)
plt.xlabel('μ')
plt.ylabel('Accuracy')

# 图例放置在图外右下角
plt.legend(loc='lower left', bbox_to_anchor=(1.02, 0), borderaxespad=0)

plt.tight_layout(rect=[0, 0, 0.85, 1])  # 给右边空出位置放图例
plt.show()