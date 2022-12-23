"""
Make a phase slice as an inset in Fig 1 - not used though.
"""

import matplotlib.pyplot as plt
import numpy as np

plt.ion()

data = np.load('rectified.npz')['data']
N = data.shape[0]
phase_offset = np.angle(data[N//2, N//2, N//2])
data = data * np.exp(-1j * phase_offset)
data[np.abs(data) < np.abs(data).max() / 5] = np.nan

fig, ax = plt.subplots()
im = ax.imshow(np.angle(data[data.shape[0] // 2]),
               cmap='jet',
               vmin=-3.14/2, vmax=3.14/2)
plt.setp(ax, 'xticks', [], 'yticks', [], 'xlim', [3,17], 'ylim', [18,4])
ax.axis('off')
cbar = fig.colorbar(im)
cbar.set_ticks([-3.14/2, 0, 3.14/2])
cbar.set_ticklabels(['$-\pi/2$', '0', '$\pi/2$'])

plt.savefig('plot2d.png', transparent=True)
