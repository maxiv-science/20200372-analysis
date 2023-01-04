""" 
Try to analyze low-potential phase transition by comparing individual
peaks.
"""

import h5py
import numpy as np
from scipy.ndimage import label
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib
import random
from utils import load_avg, radial_integral
from utils import get_potential, get_flux

matplotlib.rcParams.update({'font.size': 8})
plt.ion()

PATH = '/data/visitors/nanomax/20200372/2021062308/raw/sample/%06u.h5'
CENTER = -1500, 515//2
mask = np.load('mask.npy')


def scramble(image):
    l = list(range(image.min(), image.max()+1))
    random.shuffle(l)
    image_ = np.zeros_like(image)
    for i in range(image.min(), image.max()+1):
        image_ += (image == i) * l[i - image.min()]
    return image_

def make_mask(image, top, width):
    r = top - CENTER[0]
    ii, jj = np.indices(image.shape)
    rr = np.sqrt((ii-CENTER[0])**2 + (jj-CENTER[1])**2)
    mask = (np.abs(r - rr) <= (width / 2))
    return mask.astype(int)

def count_hits(data, plot, cutoff):
    # create a "side view" [frame-number, azimuth], with time and
    # azimuth axes binned m by n, threshold and then count hits.
    m1 = make_mask(data[0], 181, 20)
    m2 = make_mask(data[0], 117, 30)
    M, N = (data.shape[0], data.shape[-1])
    m, n = 5, 10
    # m1, first peak
    b1 = (data * m1).sum(axis=1)
    b1_ = np.reshape(b1[:M//m*m, :N//n*n], (M//m, m, N//n, n)).sum(axis=(1,3))
    cut1 = ((b1_ - np.median(b1_)) > cutoff)
    labels1, num1 = label(cut1)
    # m2, second (lower q) peak
    b2 = (data * m2).sum(axis=1)
    b2_ = np.reshape(b2[:M//m*m, :N//n*n], (M//m, m, N//n, n)).sum(axis=(1,3))
    cut2 = ((b2_ - np.median(b2_)) > cutoff)
    labels2, num2 = label(cut2)
    if plot:
        fig, ax = plt.subplots(ncols=6)
        ax[0].imshow(np.log10(b1), aspect='auto', interpolation='none')
        ax[1].imshow(cut1, aspect='auto', interpolation='none')
        ax[2].imshow(scramble(labels1), aspect='auto', cmap='gist_ncar', interpolation='none')
        ax[3].imshow(np.log10(b2), aspect='auto', interpolation='none')
        ax[4].imshow(cut2, aspect='auto', interpolation='none')
        ax[5].imshow(scramble(labels2), aspect='auto', cmap='gist_ncar', interpolation='none')
        plt.pause(.1)
    return (num1, num2)

def get_hits_for_scans(scans, plot=True):
    hitlist = []
    for scan in scans:
        print(scan)
        with h5py.File(PATH%scan, 'r') as fp:
            arr = fp['entry/measurement/merlin/frames'][:]
            arr[:] = arr * mask
        flux = get_flux(scan).mean()
        cutoff = 1000 / 2e10 * flux  # refer binned pixel signal to high-flux scan.
        hitlist.append(count_hits(arr, plot=plot, cutoff=cutoff))
    return hitlist

def integrate_scans(scans, plot=False):
    integrals = []
    for scan in scans:
        print(scan)
        with h5py.File(PATH%scan, 'r') as fp:
            arr = fp['entry/measurement/merlin/frames'][:]
            arr[:] = arr * mask
        av_image = np.mean(arr, axis=0)
        m1 = make_mask(arr[0], 181, 20)
        m2 = make_mask(arr[0], 117, 30)
        m3 = make_mask(arr[0], 150, 10)
        if plot:
            fig, ax = plt.subplots(ncols=4)
            ax[0].imshow(m1)
            ax[1].imshow(m2)
            ax[2].imshow(m3)
            ax[3].imshow(np.log10(av_image))
        integrals.append([np.mean(m1 * av_image),
                          np.mean(m2 * av_image),
                          np.mean(m3 * av_image)])
    return np.array(integrals)  # [scan, peak]

### Curve integration
if 0:
    fluxes, curves = [], []
    scans = np.arange(339, 339+15)  # first series, highest flux
    for i in range(15): # 15 fluxes
        print('***', i)
        curves_ = []
        for scan in (scans + i * 15):
            print(scan)
            pix, I = radial_integral(load_avg(scan))
            curves_.append(I)
        curves.append(curves_)
        fluxes.append(get_flux(scans[0] + i * 15).mean())
    pots = [get_potential(s) for s in scans]
    curves = np.array(curves)  # indexed [flux, pot, peak]
    np.savez('fig_flux_dependence_integration.npz', curves=curves, pots=pots, fluxes=fluxes)
else:
    dct = np.load('fig_flux_dependence_integration.npz')
    fluxes = dct['fluxes']
    pots = dct['pots']
    curves = dct['curves']

### Hit counting
if 0:
    fluxes, hits = [], []
    scans = np.arange(339, 339+15)  # first series, highest flux
    for i in range(15): # 15 fluxes
        print(scans + i * 15)
        hits.append(get_hits_for_scans(scans + i * 15, plot=False))
        pots = [get_potential(s) for s in scans]
        fluxes.append(get_flux(scans[0] + i * 15).mean())
    hits = np.array(hits)  # indexed [flux, pot, peak]
    np.savez('fig_flux_dependence_hits.npz', hits=hits, pots=pots, fluxes=fluxes)
else:
    dct = np.load('fig_flux_dependence_hits.npz')
    fluxes = dct['fluxes']
    pots = dct['pots']
    hits = dct['hits']

## Figure
fig = plt.figure(figsize=(3.33,6))
gs = GridSpec(2, 1, figure=fig, height_ratios=[1, .25])
ax = [fig.add_subplot(gs[0, 0]),
      fig.add_subplot(gs[1, 0])]
plt.subplots_adjust(bottom=.05, left=.14, right=.99, top=.99)

for iflux in range(15):
    offset = -iflux * 50
    ax[0].errorbar(pots, hits[iflux, :, 1] + offset,
                 yerr=None,  # np.sqrt(hits[iflux, :, 1]),
                 linestyle='-', marker='o', ms=2.5, lw=.8)
    flux = fluxes[iflux]
    order = int(np.log10(flux))
    prefactor = flux / (10**order)
    ax[0].text(.62, offset+10, '%.1f$\\cdot 10^{%d}$' % (prefactor, order),
             ha='right')
ax[0].text(.62, 100, 'Flux (s$^{-1}$):', ha='right')
ax[0].set_xlabel('E (V vs. Ag/AgCl sat.)')
ax[0].set_ylabel('$\\beta$-PdH$_x$(111) diffraction hits', labelpad=-5)
ax[0].set_yticks([0,100,200])


###
i0, i1 = 158, 183
p0 = curves[:, :, i0:i1].sum(axis=-1) - (curves[:, :, i0] + curves[:, :, i1]) / 2 * (i1 - i0)
i0, i1 = 95, 130
p1 = curves[:, :, i0:i1].sum(axis=-1) - (curves[:, :, i0] + curves[:, :, i1]) / 2 * (i1 - i0)
# p0 and p1 indexed [flux, pot]

kw = dict(linestyle='-', marker='o', ms=2.5, lw=.8)
i_pot = 10
p0 = p0[:, i_pot] / fluxes * 1e10
p1 = p1[:, i_pot] / fluxes * 1e10
ax[1].plot(fluxes, p0, **kw, color='k')
ax[1].plot(fluxes, p1, **kw, color='gray')
ax[1].set_xscale('log')
ax[1].set_xlabel('X-ray flux (s$^{-1}$)', labelpad=-5)
ax[1].set_ylabel('$|I|$ / $I_0$ (arb.)')
ax[1].text(8e8, p0[-1] + .4, 'Pd(111)')
ax[1].text(8e8, p1[-1] + .4, '$\\beta$-PdH$_x$(111)', color='gray')
ax[1].text(1.3e10, -.3, 'E = %.1f V' % pots[i_pot], va='bottom', ha='right')
ax[1].set_ylim([-.5, 6.3])

fig.text(.01, .98, 'a)')
fig.text(.01, .23, 'b)')
plt.savefig('fig_flux_dependence.pdf')