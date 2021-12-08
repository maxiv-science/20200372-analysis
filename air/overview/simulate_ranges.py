### Maps out diffraction of a cubic particle across a parameter
### space of phi and rocking angles.

### Defines a  particle, rotates it, and applies the
### projection operator before doing the 2D FT.

import ptypy
import nmutils
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
diameter = 24e-9

### which angles and slices to simulate
offsets = (-.5,-.25,-.1,0,.1,.25,.5)
angles=range(50, 70+1, 4)

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

### prepare q and r space plots
diff_max = 6e6
exit_max = 20.
fig, ax = plt.subplots(ncols=len(angles), nrows=len(offsets), figsize=(13, 7.55))
fig.subplots_adjust(hspace=0, wspace=0, right=.85, left=.06, bottom=.1, top=.99)
cbar_ax = fig.add_axes((.91, .2, .03, .6))
fig2, ax2 = plt.subplots(ncols=len(angles), nrows=len(offsets), figsize=(13, 7.55))
fig2.subplots_adjust(hspace=0, wspace=0, right=.85, left=.06, bottom=.1, top=.99)
cbar_ax2 = fig2.add_axes((.88, .2, .07, .6))
sbar_ax2 = fig2.add_axes((.88, .85, .07, .1))
for i in range(len(angles)):
    ax[-1, i].set_xlabel('%.0f' % angles[i])
    ax2[-1, i].set_xlabel('%.0f' % angles[i])
for i in range(len(offsets)):
    ax[i, 0].set_ylabel('%.2f' % offsets[i])
    ax2[i, 0].set_ylabel('%.2f' % offsets[i])
fig.text(.02, .55, 'rocking angle (degrees)', rotation=90, fontsize=16, ha='left', va='center')
fig2.text(.02, .55, 'rocking angle (degrees)', rotation=90, fontsize=16, ha='left', va='center')
fig.text(.5, .02, 'phi angle (degrees)', ha='center', va='bottom', fontsize=16)
fig2.text(.5, .02, 'phi angle (degrees)', ha='center', va='bottom', fontsize=16)
plt.pause(0.1)

### the main calculation loop
data = []
for iangle, angle in enumerate(angles):
    o = Cube()
    o.shift([-.5, -.5, -.5])
    o.scale(diameter)
    o.rotate('z', 45)
    o.rotate('y', 109.5/2)

    o.rotate('z', angle)
    xx, zz, yy = g.transformed_grid(S, input_space='real', input_system='natural')
    v.data[:] = o.contains((xx, -yy, zz))

    for ioffset, offset in enumerate(offsets):
        g.bragg_offset = offset

        I = np.abs(g.propagator.fw(v.data))**2
        exit = g.overlap2exit(v.data)
        data.append({'offset': offset,
                     'angle': angle,
                     'diff': I,
                     'exit': exit})

        # add shot noise and plot diffraction
        if (iangle == 0) and (ioffset == 0):
            photons_per_intensity = 1e4 / np.sum(I)
        photons = photons_per_intensity * np.sum(I)
        diff = nmutils.utils.noisyImage(I, photonsTotal=photons)
        #ax[ioffset, iangle].imshow(diff, interpolation='none', vmax=diff_max * photons_per_intensity, cmap='jet', norm=matplotlib.colors.LogNorm())
        ax[ioffset, iangle].imshow(diff, interpolation='none', cmap='jet', norm=matplotlib.colors.LogNorm())
        plt.setp(ax[ioffset, iangle], 'xticks', [], 'yticks', [])

        # plot exit waves
        midphase = np.mean(np.angle(exit[126:130, 126:130]))
        exit *= np.exp(1j * -midphase)
        phase = np.angle(exit)
        phase[np.where(np.abs(exit) < np.abs(exit).max()*.05)] = np.nan
        color_options = {'vmin':0, 'vmax':exit_max*.7, 
                         'argmin':-np.pi/4, 'argmax':np.pi/4,
                         'cmap':'hls', 'offset':.15}
        ax2[ioffset, iangle].imshow(nmutils.utils.complex2image(exit, **color_options), interpolation='none')
#        ax2[ioffset, iangle].imshow(np.abs(exit))
        plt.setp(ax2[ioffset, iangle],
            'xlim', [128-14, 128+14], 'ylim', [128+14, 128-14],
            'xticks', [], 'yticks', [])

        plt.draw()
        plt.pause(.01)

for a in ax.flatten(): 
    a.set_ylim(85, 255-85) 
    a.set_xlim(85, 255-85) 
