"""
Rectifies and resamples the full reconstruction on a regular grid
with aspect ratio 1:1:1, for easier visualization.
"""

import matplotlib.pyplot as plt
import numpy as np
import h5py
from bcdiass.utils import rectify_sample
import os
plt.ion()

theta = 19.

def make_square(a, n):
    """
    Take an array and make it equal in all dimensions (namely n)
    by padding or cropping.
    """
    for dim in range(a.ndim):
        add = n - a.shape[dim]
        if add > 0:
            pad = [(0, 0),] * a.ndim
            before, after = add // 2, add - add // 2
            pad[dim] = (before, after)
            a = np.pad(a, pad)
        elif add < 0:
            first = -add // 2
            a = np.take(a, indices=(range(first, first + n)), axis=dim)
    return a

# load the q space scales and work out the new q range after cropping etc
Q3, Q1, Q2 = np.load('assembled.npz')['Q'] # full range of original assembly
nq3, nq1, nq2 = np.load('assembled.npz')['W'].shape # original shape
dq3, dq1, dq2 = Q3/nq3, Q1/nq1, Q2/nq2 # original q space pixel sizes
N_recons = np.load('prepared.npz')['data'].shape
Q3, Q1, Q2 = np.array((dq3, dq1, dq2)) * N_recons # full q range used in the reconstruction

Q3 *= 3  # there is an ambiguity in the assembly - this is a free parameter.

# resolution: res * qmax = 2 pi
#   - if qmax is half the q range (origin to edge) then res is the full period resolution
#   - if qmax is the full q range (edge to edge) then res is the pixel size
dr3, dr1, dr2 = (2 * np.pi / q for q in (Q3, Q1, Q2)) # half-period res (pixel size)
with h5py.File('modes.h5', 'r') as fp:
    p = fp['entry_1/data_1/data'][0]
nr3, nr1, nr2 = p.shape


### plot along all three axes to understand the aspect ratio
fig, ax = plt.subplots(ncols=3)
intp = {'interpolation': 'none'}

# from the front
ext = np.array((-dr2*nr2/2, dr2*nr2/2, -dr1*nr1/2, dr1*nr1/2)) * 1e9
ax[0].imshow(np.abs(p).sum(axis=0), extent=ext, **intp)
plt.setp(ax[0], xlim=[-50,50], ylim=[-50,50], title='front view',
         xlabel='r2', ylabel='r1')
# from the top
ext = np.array((-dr2*nr2/2, dr2*nr2/2, -dr3*nr3/2, dr3*nr3/2)) * 1e9
im = np.flip(np.abs(p).sum(axis=1), axis=0)
ax[1].imshow(im, extent=ext, **intp)
plt.setp(ax[1], xlim=[-50,50], ylim=[-50,50], title='top view',
         xlabel='r2', ylabel='r3')
# from the side
ext = np.array((-dr3*nr3/2, dr3*nr3/2, -dr1*nr1/2, dr1*nr1/2)) * 1e9
im = np.transpose(np.abs(p).sum(axis=2))
ax[2].imshow(im, extent=ext, **intp)
plt.setp(ax[2], xlim=[-50,50], ylim=[-50,50], title='side view',
         xlabel='r3', ylabel='r1')
fig.suptitle('2d projections')


### rectify and plot
p, psize = rectify_sample(p, (dr3, dr1, dr2), theta)

p = make_square(p, min(p.shape) + 4)
half = p.shape[0] * psize / 2
extent = (-half, half) * 2
fig, ax = plt.subplots(ncols=3)

ax[0].imshow(np.abs(p).sum(axis=0), extent=extent, **intp)
ax[1].imshow(np.abs(p).sum(axis=1), extent=extent, **intp)
ax[2].imshow(np.abs(p).sum(axis=2), extent=extent, **intp)

np.savez('rectified.npz', psize=(dr1, dr2, dr3), data=p)

plt.show()
