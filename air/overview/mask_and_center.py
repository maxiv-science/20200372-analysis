"""
Add up frames to estimate the powder rings' center.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
#plt.ion()

scan = 234

fn = 'scan%u_sum.npz'%scan
if os.path.exists(fn):
    data = np.load(fn)['data'].astype(np.int64)
else:
    filename = '/data/visitors/nanomax/20200372/2021062308/raw/sample/%06u.h5' % scan
    with h5py.File(filename, 'r') as fp:
        dset = fp['entry/measurement/merlin/frames']
        data = np.zeros(dset.shape[1:], dtype=np.int64)
        for i in range(dset.shape[0] // 1000):
            print(i)
            data += dset[i*1000:(i+1)*1000].sum(axis=0)
    np.savez(fn, data=data)

# mask
mask_file = '/data/visitors/nanomax/common/masks/merlin/latest.h5'
with h5py.File(mask_file, 'r') as fp:
    old_mask = fp['mask'][:]
new_mask = ((data < 10**5.5) & (data > 1e3)).astype(int)
mask = new_mask * old_mask
np.savez('mask.npz', mask=mask)
data[np.where(mask==0)] = -1

# center of an approximate circle
y0, x0 = -1700, 256
x = np.arange(0, 515)
def y(ring):
    r = -y0 + ring
    return np.sqrt(r**2 - (x-x0)**2) + y0

plt.figure()
plt.imshow(data, vmax=data.max()/5)
plt.plot(x, y(92), 'r')
plt.plot(x, y(353), 'r')
