import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib
from utils import (load_avg, get_flux, get_potential, mask, PATH,
                   radial_integral)

matplotlib.rcParams.update({'font.size': 8})

plt.ion()

# data to compare:
# 234 - in air                   - ring at pixel 93
# 247 - acid, same detector pos  - ring at pixel 83 (0.55 V)
# 248 - acid, moved the detector - ring at pixel 181 (0.55 V)
# 248 - acid, moved the detector - beta rings between 110 and 125

# with radial integration:
# 234 - in air                   - ring at pixel 95.5
# 247 - acid, same detector pos  - ring at pixel 83.5 (0.55 V)
# 248 - acid, moved the detector - ring at pixel 180 (0.55 V)
# 248 - acid, moved the detector - beta rings between 110 and 130

# detector distance around 120 mm, we'll use Pd(111) and Pd(222)
# for each loading (in air, in the cell, then after moving the
# detector) for calibration. The details of the cell mounting
# affects the apparent angles.

# geometry input
lam = 12.4 / 8.5 * 1e-10
rad_per_pix = 55e-6 / .131
a_Pd = 3.887e-10  # Manchester 1994
a_PdH = 4.025e-10 #   "
twoth_Pd111 = 2 * np.arcsin(lam / (2 * a_Pd / np.sqrt(3)))
shift = 180 - 83.5

# work out d-spacings each ring, based on where the top of the ring
# meets the horizontal center of the detector. Before and after moving
# the detector.
d_before = lam / 2 / np.sin(
    (twoth_Pd111 + (np.arange(515) - 95.5) * rad_per_pix) / 2)
d_after = lam / 2 / np.sin(
    (twoth_Pd111 + (np.arange(515) - 180) * rad_per_pix*1.045) / 2)

### now for the plotting
plt.close('all')
fig = plt.figure(figsize=(3.33,4))
gs = GridSpec(5, 3, figure=fig, width_ratios=[1, 1, 3])
ax0 = fig.add_subplot(gs[:, 2])
# fig, ax = plt.subplots(ncols=2, nrows=1)

### first some powder plots
# 234: Pd in air before moving the detector
pix, I234 = radial_integral(load_avg(234))
d234 = d_before[pix] * 1e10
# #247: Pd in acid before moving the detector (sanity check)
pix, I247 = radial_integral(load_avg(247))
d247 = d_before[pix] * 1e10
# 339: 0.6V after moving detector
pix, I339 = radial_integral(load_avg(339))
d339 = d_after[pix] * 1e10
# 349: 0.1V after moving detector
pix, I349 = radial_integral(load_avg(349))
d349 = d_after[pix] * 1e10
# 549: 0.5V at low flux
pix, I549 = radial_integral(load_avg(549))
d549 = d_after[pix] * 1e10
###
ax0.plot(d234, I234 * 4 - .3, label='in air')
ax0.plot(d339, I339, label='%.1f V'%get_potential(339))
# ax0.plot(d247, I247, label='before move')
ax0.plot(d349, I349, label='%.1f V'%get_potential(349))
# ax0.plot(d549, I549*50, label='low flux, %.1f V'%get_potential(549))
ax0.set_xlim([2.2, 2.4])
ax0.axvline(a_Pd * 1e10 / np.sqrt(3), linestyle='--', color='k')
ax0.axvline(a_PdH * 1e10 / np.sqrt(3), linestyle='--', color='k')
ax0.legend(loc='lower right')
ax0.set_title('c) (111) intensity', loc='left', fontsize=8)
ax0.text(-.01 + a_Pd * 1e10 / np.sqrt(3), 2.2, 'Pd ', ha='right')
ax0.text(.01 + a_PdH * 1e10 / np.sqrt(3), 2.2, '$\\beta$-PdH$_x$', ha='left')
ax0.set_xlabel('d / Å')
ax0.set_yticks([])

pars = dict(aspect='auto', interpolation='none')

### then some images - here's air
ax1 = [fig.add_subplot(gs[i, 0]) for i in range(5)]
i0, i1 = 63073+5, 63136-10
with h5py.File(PATH % 235, 'r') as fp:
    for i in range(5):
        im = fp['entry/measurement/merlin/frames'][i0+(i1-i0)//5*i]
        im = im[:80, 210:390]
        ii, jj = np.indices(im.shape)
        com = np.sum(jj * im) / np.sum(im)
        im = np.roll(im, -int(com - im.shape[1]//2), axis=1)
        cut = (im.shape[1] - im.shape[0]) // 2
        im = im[:, cut:-cut]
        ax1[i].imshow(np.log10(im), vmax=3, **pars)
        plt.setp(ax1[i], xticks=[], yticks=[])
ax1[0].set_title('a) air', loc='left', fontsize=8)

### then some more images - here's acid
ax2 = [fig.add_subplot(gs[i, 1]) for i in range(5)]
frames = [353, 355, 356, 357, 358]
R = 1000
x = np.linspace(0, 120)
y = np.sqrt(R**2 - (x-60)**2) - R
with h5py.File(PATH % 351, 'r') as fp:
    for i in range(5):
        im = fp['entry/measurement/merlin/frames'][frames[i]]
        im = im[100:220, 150:350]
        ii, jj = np.indices(im.shape)
        com = np.sum(jj * im) / np.sum(im)
        #im = np.roll(im, -int(com - im.shape[1]//2), axis=1)
        cut = (im.shape[1] - im.shape[0]) // 2
        im = im[:, cut:-cut]
        ax2[i].imshow(np.log10(im), origin='lower', vmax=2, **pars)
        ax2[i].plot([], []) # for the colors
        ax2[i].plot(x, y + 77, '--')
        ax2[i].plot(x, y + 27, '--')
        plt.setp(ax2[i], xticks=[], yticks=[])
ax2[0].set_title('b) acid', loc='left', fontsize=8)

### add an inset with a reconstruction
a = fig.add_axes([.03, .0, .2, .18])
a.set_xticks([])
a.set_yticks([])
a.axis('off')
im = plt.imread('../air/assembly/235_63/plot3d.png')
alpha = (im.sum(axis=2) < 2.999).astype(float)
im = np.pad(im, ((0, 0), (0, 0), (0, 1)))
im[:, :, -1] = alpha
a.imshow(im)

plt.subplots_adjust(bottom=.12, left=.01, right=.95, top=.95, wspace=.1, hspace=.05)
plt.savefig('fig_diffraction.pdf')
