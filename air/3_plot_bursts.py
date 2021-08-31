"""
This script shows animated movies of spontaneous rocking curves,
using hits found by a preliminary search. The hit indices are
only examples, and by no means exhaustive.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt

# list of strongest tentative (111) hits
#scan, burst = (234, 7) # very nice
#scan, burst = (234, 32)
#scan, burst = (234, 53)
#scan, burst = (235, 59)
scan, burst = (235, 63) # very nice
#scan, burst = (236, 21)
#scan, burst = (236, 92)
#scan, burst = (236, 93)
#scan, burst = (238, 100)
#scan, burst = (238, 107)

# list of strongest tentative (200) hits
#scan, burst = (234, 117)
#scan, burst = (234, 133) # very nice
#scan, burst = (235, 92) # very nice
#scan, burst = (235, 123)
#scan, burst = (236, 13)
#scan, burst = (236, 49)
#scan, burst = (236, 117)
#scan, burst = (238, 131)
#scan, burst = (239, 49) # twins? cool


Nb = 1000
inpath = '/data/visitors/nanomax/20200372/2021062308/raw/sample/'
pattern = 'scan_%06u_merlin.hdf5'
hdfpath = 'entry/measurement/Merlin/data'
maskfile = '/data/visitors/nanomax/common/masks/merlin/latest.h5'

with h5py.File(maskfile, 'r') as fp:
    mask = fp['mask'][:]

with h5py.File(inpath+pattern%scan, 'r') as fp:
    data = fp[hdfpath][burst*Nb:(burst+1)*Nb,:,:]
data[:] = data * mask

#fig, ax = plt.subplots(nrows=2, )
fig, ax = plt.subplots(nrows=2, gridspec_kw={'height_ratios':[.4,1]}, figsize=(7,12)) 
plt.subplots_adjust(hspace=.1, top=.99, bottom=.1, right=.95, left=.05)
ax[0].plot(data.max(axis=(1,2)))

l = ax[0].axvline(0)
data = np.log10(data)
center = np.argmax(data.max(axis=(1,2)))
begin = max(0, center-30)
end = min(data.shape[0]-1, center+100)
mx = data.max()
for i in range(begin, end):
    ax[1].clear()
    ax[1].imshow(data[i], vmax=mx, cmap='jet')
    l.remove()
    l = ax[0].axvline(i)
    ax[0].set_title('frame %u'%i)
    plt.pause(.001)
    plt.savefig('dumps/%u_%u_%u.png'%(scan, burst, i))
