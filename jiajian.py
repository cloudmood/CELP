import matplotlib as mpl
mpl.use('tkagg')


import matplotlib.pyplot as plt
import numpy as np

# 横坐标标签
datasets = ['Cora', 'Citeseer', 'Pubmed']

# 平均值和标准差数据
mean_std_data = {
    'w/o add':    ([92.41, 93.70, 83.41], [1.06, 1.83, 1.64]),
    'w/o remove': ([92.82, 94.78, 83.94], [0.59, 0.94, 1.81]),
    'CELP':       ([93.34, 95.41, 84.11], [0.54, 0.69, 1.57]),
}

# 颜色设定
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# 柱宽 & 横坐标位置
bar_width = 0.2
x = np.arange(len(datasets))

# 画布
plt.figure(figsize=(8, 6))

# 绘图
for i, ((label, (means, stds)), color) in enumerate(zip(mean_std_data.items(), colors)):
    offset = (i - 1) * bar_width  # 居中排列 3 组柱子
    plt.bar(x + offset, means, yerr=stds, capsize=3, width=bar_width, label=label, color=color)

# 坐标轴设置
plt.ylabel('Accuracy', fontsize=12)
plt.xticks(x, datasets, fontsize=12)
plt.ylim(80, 100)


# 图例放图内左上角
plt.legend(loc='upper left', fontsize=10, frameon=True)

plt.tight_layout()
plt.show()