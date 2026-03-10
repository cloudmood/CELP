
import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np

mu = np.array([1, 3, 5, 7, 9])

data_hr50 = {
    'Photo':    np.array([57.20, 57.64, 58.13, 58.75, 58.66]),
    'Computers':np.array([43.35, 43.30, 43.64, 43.84, 43.18]),
    'Collab':   np.array([66.02, 66.16, 66.95, 66.59, 66.27]),
}

mu_scaled_hr50 = {
    'Photo': mu * 12,
    'Computers': mu * 12,
    'Collab': mu * 12,
}

plt.figure(figsize=(8,5))

for dataset in data_hr50:
    plt.plot(mu_scaled_hr50[dataset], data_hr50[dataset], marker='o', label=dataset)

all_ticks = np.unique(np.concatenate([mu_scaled_hr50[d] for d in mu_scaled_hr50]))
plt.xticks(all_ticks)
plt.gca().set_xticklabels([f'{int(t)}' for t in all_ticks])
plt.tick_params(axis='x', direction='in', length=6)

plt.grid(False)

plt.xlabel('K')
plt.ylabel('HR@50')
plt.legend()
plt.tight_layout()
plt.show()
