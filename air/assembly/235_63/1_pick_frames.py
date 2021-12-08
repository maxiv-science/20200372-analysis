"""
Visualizes a single burst from a given scan as a video.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
#plt.ion()

scan, pos = 235, 63
begin, end = 63, 143 #178
burst_n = 1000
shape = 160

filename = '/data/visitors/nanomax/20200372/2021062308/raw/sample/%06u.h5' % scan

# load and mask data
with h5py.File(filename, 'r') as fp:
    burst = fp['entry/measurement/merlin/frames'][pos*burst_n:(pos+1)*burst_n].astype(np.int32) # to allow negative masked values
    burst = burst[begin:end]
mask = np.load('../../overview/mask.npz')['mask']
masked = np.where(mask == 0)
burst[:, masked[0], masked[1]] = -1

# find the brightest frame and its center
brightest = np.unravel_index(np.argmax(burst), burst.shape)[0]
center = np.unravel_index(np.argmax(burst[brightest]), burst[brightest].shape)

# crop the whole burst around this pixel
before_i = -min(center[0] - shape//2, 0)
before_j = -min(center[1] - shape//2, 0)
after_i = max(center[0] + shape//2 - burst.shape[-2], 0)
after_j = max(center[1] + shape//2 - burst.shape[-1], 0)
burst = np.pad(burst, 
               pad_width=((0, 0), (before_i, after_i), (before_j, after_j)),
               mode='constant', constant_values=0)
burst = burst[:,
              before_i+center[0]-shape//2:before_i+center[0]+shape//2,
              before_j+center[1]-shape//2:before_j+center[1]+shape//2]
np.savez('picked.npz', data=burst, center=center)

fig, ax = plt.subplots(ncols=2)
ax[0].imshow(np.log10(burst[brightest]))
ax[1].plot(np.max(burst, axis=(1,2)))

plt.show()
