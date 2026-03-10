import matplotlib as mpl
mpl.use('tkagg')

import matplotlib.pyplot as plt
import numpy as np


layers = [2, 4, 8, 16, 32, 64]
hits_ge = [93.34, 92.99, 92.61, 93.08, 92.99, 90.67]
hits_no_ge = [88.34, 86.72, 86.54, 83.70, 82.69, 63.60]

plt.figure(figsize=(8, 5))
plt.xscale('log')  # 关键步骤：设置横坐标为对数刻度
plt.plot(layers, hits_ge, marker='o', label='Cora w GE', linewidth=2)
plt.plot(layers, hits_no_ge, marker='s', label='Cora w/o GE', linewidth=2)

plt.xticks(layers, labels=[str(l) for l in layers])  # 显示所有层数为刻度
plt.xlabel('Number of Layers')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
