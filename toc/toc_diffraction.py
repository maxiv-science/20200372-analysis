"""
Show a number of frames for the TOC graphic.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

PATH = '/data/visitors/nanomax/20200372/2021062308/raw/sample/%06u.h5'

frames = [353, 354, 355, 356, 357, 358]
fig, ax = plt.subplots(ncols=len(frames), figsize=(5.6, 1))
fig.subplots_adjust(wspace=.05, hspace=0, right=1., left=.0, bottom=.0, top=1.)
R = 1000
x = np.linspace(0, 120)
y = np.sqrt(R**2 - (x-60)**2) - R
with h5py.File(PATH % 351, 'r') as fp:
    for i in range(len(frames)):
        im = fp['entry/measurement/merlin/frames'][frames[i]]
        im = im[100:220, 150:300]
        cut = (im.shape[1] - im.shape[0]) // 2
        im = im[:, cut:-cut]
        ax[i].imshow(np.log10(im), origin='lower', vmax=1.5)
        ax[i].plot(x, y + 77, '--', color='red')
        ax[i].plot(x, y + 27, '--', color='red')
        plt.setp(ax, xticks=[], yticks=[])
        ax[i].set_xlim(10, 110)
        ax[i].set_ylim(0, 110)

plt.savefig('toc_diffraction.png', dpi=600)
