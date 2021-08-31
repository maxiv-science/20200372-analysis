### Like the simulation script, but directly compares an
### experimental pattern with the simulated equivalent.
### Result: 45 nm.

import ptypy
import nmutils
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches
plt.ion()
from nmutils.utils.bodies import Cube

try:
    ptypy.core.geometry_bragg.Geo_BraggProjection
except AttributeError:
    raise Exception('Use 3dBPP ptypy version!')

# physics
a = 3.92e-10
#E = 8500.
E = 8500
d = a / np.sqrt(3) # (111)
hc = 4.136e-15 * 3.000e8
theta = np.arcsin(hc / (2 * d * E)) / np.pi * 180
psize = 55e-6
distance = .120
#diameter = 24e-9
diameter = 45e-9

### which angles and slices to simulate
offsets = (-.5,-.25,-.1,0,.1,.25,.5)
angles=range(50, 70+1, 4)

angle = 70
offset = -.15

### Define a Bragg geometry
g = ptypy.core.geometry_bragg.Geo_BraggProjection(psize=(psize, psize),
    shape=(51, 256, 256), energy=E*1e-3, distance=distance, theta_bragg=theta,
    bragg_offset=0.0, r3_spacing=1e-9)

### make the object container and storage
C = ptypy.core.Container(data_type=np.complex128, data_dims=3)
pos = [0, 0, 0]
pos_ = g._r3r1r2(pos)
v = ptypy.core.View(C, storageID='Sobj', psize=g.resolution, coord=pos_, shape=g.shape)
S = C.storages['Sobj']
C.reformat()

### calculate

o = Cube()
o.shift([-.5, -.5, -.5])
o.scale(diameter)
o.rotate('z', 45)
o.rotate('y', 109.5/2)
o.rotate('z', angle)
xx, zz, yy = g.transformed_grid(S, input_space='real', input_system='natural')
v.data[:] = o.contains((xx, -yy, zz))
g.bragg_offset = offset
I = np.abs(g.propagator.fw(v.data))**2
exit = g.overlap2exit(v.data)

# add shot noise and plot diffraction
photons_per_intensity = 1e5 / np.sum(I)
photons = photons_per_intensity * np.sum(I)
diff = nmutils.utils.noisyImage(I, photonsTotal=photons)

# plot
w = 40
fig, ax = plt.subplots(nrows=2)
ax[0].imshow(diff, interpolation='none', cmap='jet', norm=matplotlib.colors.LogNorm())
ax[0].set_xlim(128-w//2, 128+w//2)
ax[0].set_ylim(128-w//2, 128+w//2)
ax[0].set_title('simulation with d=%.0f nm'%(diameter*1e9))

# real data
with h5py.File('/data/visitors/nanomax/20200372/2021062308/raw/sample/000234.h5', 'r') as fp:
    im = fp['entry/measurement/merlin/frames'][7691]
ax[1].imshow(np.log10(im), vmax=np.log10(2000), cmap='jet')
ax[1].set_ylim(82-w//2, 82+w//2)
ax[1].set_xlim(38-w//2, 38+w//2)
