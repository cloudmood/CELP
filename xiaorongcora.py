import matplotlib as mpl
mpl.use('tkagg')

import matplotlib.pyplot as plt
import numpy as np

# 横轴：数据集
datasets = ['Cora', 'Citeseer', 'Pubmed']

# 只保留 4 个方法的数据
mean_std_data = {
    'w/o SE': ([91.77, 92.21, 82.61], [0.89, 1.81, 1.51]),
    'w/o LE': ([91.93, 91.35, 83.09], [1.11, 1.39, 1.56]),
    'w/o GE': ([88.34, 88.60, 78.59], [1.48, 1.54, 1.52]),
    'CELP':   ([92.41, 93.70, 83.41], [1.06, 1.83, 1.64]),
}
# 设置颜色
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# 柱状图参数
bar_width = 0.18
x = np.arange(len(datasets))

# 绘图
plt.figure(figsize=(8, 6))

for i, ((label, (means, stds)), color) in enumerate(zip(mean_std_data.items(), colors)):
    offset = (i - 1.5) * bar_width  # 居中对齐4根柱子
    plt.bar(x + offset, means, yerr=stds, capsize=3, width=bar_width, label=label, color=color)

# 设置轴与标题
plt.ylabel('HR@100', fontsize=12)
plt.xticks(x, datasets, fontsize=12)
plt.ylim(70, 100)

# 图例放在图内左上角
plt.legend(loc='upper right', fontsize=10, frameon=True)

plt.tight_layout()
plt.show()
