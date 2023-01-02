""" 
Try to analyze low-potential phase transition by comparing individual
peaks.
"""

import h5py
import numpy as np
from scipy.ndimage import label
import matplotlib.pyplot as plt
import matplotlib
import random

from utils import get_potential, get_flux

matplotlib.rcParams.update({'font.size': 8})


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

if True:
    fluxes, hits = [], []
    scans = np.arange(339, 339+15)  # first series, highest flux
    for i in range(15): # 15 fluxes
        print(scans + i * 15)
        hits.append(get_hits_for_scans(scans + i * 15, plot=False))
        pots = [get_potential(s) for s in scans]
        fluxes.append(get_flux(scans[0] + i * 15).mean())
    hits = np.array(hits)  # indexed [flux, pot, peak]
    np.savez('fig_flux_dependence.npz', hits=hits, pots=pots, fluxes=fluxes)
else:
    dct = np.load('fig_flux_dependence.npz')
    fluxes = dct['fluxes']
    pots = dct['pots']
    hits = dct['hits']

plt.figure(figsize=(3.33,5))
plt.subplots_adjust(bottom=.08, left=.16, right=.99, top=.99)
for iflux in range(15):
    offset = -iflux * 50
    plt.plot(pots, hits[iflux, :, 1] + offset, 'x-',)# lw=2-iflux/10)
    flux = fluxes[iflux]
    order = int(np.log10(flux))
    prefactor = flux / (10**order)
    plt.text(.6, offset+10, '%.1f $\\times 10^{%d}$' % (prefactor, order),
             ha='right')
plt.text(.6, 100, 'Flux (s$^{-1}$):', ha='right')
plt.xlabel('E (V vs. Ag/AgCl sat.)')
plt.ylabel('$\\beta$-PdH$_x$(111) intensity')
plt.yticks([0,100,200])
plt.savefig('fig_flux_dependence.pdf')
