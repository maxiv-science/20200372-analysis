"""
Show a number of frames for the TOC graphic.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

PATH = '/data/visitors/nanomax/20200372/2021062308/raw/sample/%06u.h5'

frames = [353, 354, 355, 356, 357, 358, 359, 360]
fig, ax = plt.subplots(ncols=len(frames), figsize=(7, 1))
fig.subplots_adjust(wspace=.02, hspace=0, right=.99, left=.01)
R = 1000
x = np.linspace(0, 120)
y = np.sqrt(R**2 - (x-60)**2) - R
with h5py.File(PATH % 351, 'r') as fp:
    for i in range(len(frames)):
        im = fp['entry/measurement/merlin/frames'][frames[i]]
        im = im[100:220, 150:350]
        cut = (im.shape[1] - im.shape[0]) // 2
        im = im[:, cut:-cut]
        ax[i].imshow(np.log10(im), origin='lower', vmax=1.5)
        ax[i].plot(x, y + 77, '--', color='gray')
        ax[i].plot(x, y + 27, '--', color='gray')
        plt.setp(ax, xticks=[], yticks=[])
        ax[i].set_xlim(0, 120)
        ax[i].set_ylim(0, 120)

plt.savefig('toc_diffraction.png', dpi=600)
