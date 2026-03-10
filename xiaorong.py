import matplotlib as mpl
mpl.use('tkagg')

import matplotlib.pyplot as plt
import numpy as np

# 数据集
datasets = ['Photo', 'Computers', 'Collab']

# 只保留 4 个方法的数据
mean_std_data = {
    'w/o SE': ([57.41, 42.21, 65.97], [3.74, 3.70, 0.42]),
    'w/o LE': ([52.72, 41.35, 65.46], [3.33, 2.95, 0.62]),
    'w/o GE': ([58.07, 42.71, 66.09], [2.81, 3.12, 0.71]),
    'CELP':   ([58.75, 43.84, 66.95], [3.91, 2.59, 0.63]),
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
plt.ylabel('HR@50', fontsize=12)
plt.xticks(x, datasets, fontsize=12)
plt.ylim(35, 75)

# 图例放在图内左上角
plt.legend(loc='upper left', fontsize=10, frameon=True)

plt.tight_layout()
plt.show()
