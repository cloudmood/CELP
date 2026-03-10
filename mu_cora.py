import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np

# μ 和 HR@100 平均值（忽略 ± 部分）
mu = np.array([0, 0.05, 0.15, 0.25, 0.5, 0.75])

data = {
    'Cora':     np.array([91.77, 92.41, 91.56, 91.00, 89.57, 89.29]),
    'Citeseer': np.array([92.21, 93.70, 91.71, 90.88, 90.22, 89.24]),
    'Pubmed':   np.array([82.61, 83.41, 82.95, 82.05, 81.16, 80.32]),
}

plt.figure(figsize=(8,5))

for dataset in data:
    plt.plot(mu, data[dataset], marker='o', label=dataset)

# 设置横坐标以 0.25 为间隔划分
xticks = np.arange(0, 1.0, 0.25)
plt.xticks(xticks)
plt.gca().set_xticklabels([f'{x:.2f}' for x in xticks])
plt.tick_params(axis='x', direction='in', length=6)

plt.grid(False)
plt.xlabel('μ')
plt.ylabel('HR@100')
plt.legend()
plt.tight_layout()
plt.show()
