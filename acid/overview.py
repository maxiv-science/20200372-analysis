import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from nmutils.utils.ion_chamber import Ionchamber
plt.ion()
plt.close('all')

PATH = '/data/visitors/nanomax/20200372/2021062308/raw/sample/%06u.h5'
CENTER = -1500, 515//2
mask = np.load('mask.npy')


def load_avg(scannr):
    npy = '%06u.npy' % scannr
    if os.path.exists(npy):
        tot = np.load(npy) * mask
    else:
        with h5py.File(PATH % scannr, 'r') as fp:
            dset = fp['entry/measurement/merlin/frames']
            tot = np.zeros(dset[0].shape, dtype=np.uint64)
            N = 1000
            for i in range(dset.shape[0] // N):
                chunk = np.sum(dset[i*N:(i+1)*N], axis=0)
                tot[:] = tot + chunk
            tot = tot / dset.shape[0]
        np.save(npy, tot)
    tot[:] = tot * mask
    return tot


def get_flux(scannr):
    ic = Ionchamber(1.5)
    with h5py.File(PATH % scannr, 'r') as fp:
        current = fp['entry/measurement/alba2/1'][:]
        energy = fp['entry/snapshot/energy'][0]
    return ic.flux(current, energy)


def get_potential(scannr):
    with h5py.File(PATH % scannr, 'r') as fp:
        pot = fp['entry/snapshot/pot'][0]
    return pot


def radial_integral(image):
    tops = np.arange(50, 500) # tops of the rings on the detector
    r = tops - CENTER[0]
    ii, jj = np.indices(image.shape)
    rr = np.sqrt((ii-CENTER[0])**2 + (jj-CENTER[1])**2)
    avgs = []
    for r_ in r:
        mask = (np.abs(r_ - rr) <= .5)
        avgs.append(np.mean(image[mask]))
    return tops, np.array(avgs)


### lattice parameter in air - before moving the detector.
if 0:
    scan = 234
    data = load_avg(scan)
    data[data > 2] = 0
    av_flux = get_flux(scan).mean()
    fig, ax = plt.subplots(ncols=2)
    ax[0].imshow(data, origin='lower')
    ax[1].plot(data.sum(axis=1))
    fig.suptitle('%u in air for reference (%.1e ph/s)' % (scan, av_flux))


### 247 flyscan lines at different potentials
### before detector move
if 0:
    scan = 247
    with h5py.File(PATH % scan, 'r') as fp:
        data = fp['entry/measurement/merlin/frames'][:]
        data = data.reshape((201, -1, 515, 515)) * mask.reshape((1, 1, 515, 515))
    av_flux = get_flux(scan).mean()
    fig, ax = plt.subplots(ncols=2)
    ax[0].imshow(data.mean(axis=(0,1)), origin='lower')
    for i in range(data.shape[1]):
        ax[1].plot(data[:, i, :, :].sum(axis=(0,-1)))
    plt.suptitle('%u - 11 potentials, detector pos. same as 234 etc (%.1e ph/s)' % (scan, av_flux))


### 248: flyscan lines at different potentials
### after detector move. 4.3e-8 A ~ 7e10 ph/s
if 0:
    scan = 248
    with h5py.File(PATH % scan, 'r') as fp:
        data = fp['entry/measurement/merlin/frames'][:]
        data = data.reshape((201, -1, 515, 515)) * mask.reshape((1, 1, 515, 515))
    av_flux = get_flux(scan).mean()
    fig, ax = plt.subplots(ncols=2)
    ax[0].imshow(data.mean(axis=(0,1)), origin='lower')
    for i in range(data.shape[1]):
        ax[1].plot(data[:, i, :, :].sum(axis=(0,-1)))
    plt.suptitle('%u - 11 potentials again, moved det. (%.1e ph/s)' % (scan, av_flux))


def potential_dependence(scans, show_images=True):
    """
    Load a series of averaged scans, integrate over p1 and p2 for both
    111 and 200 rings.
    """
    if show_images:
        fig, ax = plt.subplots(ncols=4, nrows=4)
        ax = ax.flatten()
    fig2, ax2 = plt.subplots(nrows=3)
    pots, p1, p2, p1_, p2_ = [], [], [], [], []
    for i, scan in enumerate(scans):
        avg = load_avg(scan)
        pot = get_potential(scan)
        if show_images:
            ax[i].imshow(avg, origin='lower', vmax=4)
            ax[i].set_title('%u: %.2f V' % (scan, pot))
        r, I = radial_integral(avg)
        # 111
        norm = I  / I[20:50].mean()
        ax2[0].plot(norm, label=pot)
        bg = np.array((50, 100, 145))
        p1.append(np.sum(norm[bg[0]:bg[1]] - norm[bg[:2]].mean()))
        p2.append(np.sum(norm[bg[1]:bg[2]] - norm[bg[1:]].mean()))
        # 200
        norm_ = I  / I[330:355].mean()
        ax2[1].plot(norm_, label=pot)
        bg_ = np.array((289, 325, 360, 390))
        p1_.append(np.sum(norm_[bg_[0]:bg_[1]] - norm_[bg_[:2]].mean()))
        p2_.append(np.sum(norm_[bg_[2]:bg_[3]] - norm_[bg_[2:]].mean()))
        pots.append(pot)
    ax2[0].plot(bg[:2], norm[bg[:2]], 'kx--')
    ax2[0].plot(bg[1:], norm[bg[1:]], 'kx--')
    ax2[1].plot(bg_[:2], norm_[bg_[:2]], 'kx--')
    ax2[1].plot(bg_[1:], norm_[bg_[1:]], 'kx--')
    twin = ax2[2].twinx()
    p1 = np.array(p1) + np.array(p1_)
    p2 = np.array(p2) + np.array(p2_)
    twin.plot(pots, p1, 'rx')
    twin.set_ylabel('p1, low-q, x:s (111+200)')
    twin.set_ylim([-np.ptp(twin.get_ylim())*.1, twin.get_ylim()[1]])
    ax2[2].plot(pots, p2, 'ko')
    ax2[2].set_ylim([-np.ptp(ax2[2].get_ylim())*.1, ax2[2].get_ylim()[1]])
    ax2[2].set_ylabel('p2, high-q, o:s (111+200)')
    ax2[0].legend()
    f1 = get_flux(scans[0]).mean()
    f2 = get_flux(scans[1]).mean()
    title = '#%u-%u (%.1e ph/s)' % (scans[0], scans[-1], (f1+f2)/2)
    if show_images:
        fig.suptitle(title)
    fig2.suptitle(title)
    return pots, p1, p2

if 0:
    # 323-337: first reversible potential series, 1.2e10
    potential_dependence(np.arange(323, 337+1), show_images=True)


if 0:
    # 339-564 decreasing flux, parameter overview
    plt.figure()
    scans = np.arange(339, 564+1)
    fluxes, pots = [], []
    for scan in scans:
        avg = load_avg(scan)
        fluxes.append(get_flux(scan).mean())
        pots.append(get_potential(scan))
    plt.plot(scans, np.log10(fluxes))
    plt.gca().twinx().plot(scans, pots, 'r.-')
    plt.title('#339-564: log flux and potential')


if 0:
    # find center and radii for the two 111 peaks
    avg = load_avg(339)
    fig, ax = plt.subplots(ncols=3)
    ax[0].imshow(avg, origin='lower')
    r111 = -CENTER[0] + 180
    r200 = -CENTER[0] + 425
    ii, jj = np.indices(avg.shape)
    radii = np.sqrt((ii - CENTER[0])**2 + (jj - CENTER[1])**2)
    mask111 = (np.abs(radii - r111) < 5)
    ax[1].imshow(avg * .1 + (avg * mask111) * .9, origin='lower')
    mask200 = (np.abs(radii - r200) < 5)
    ax[2].imshow(avg * .1 + (avg * mask200) * .9, origin='lower')

if 0:
    # potential dependence for all fluxes, plotting the rise of the
    # low-q peak in the average data.
    Npots = 15
    Nfluxes = 15
    s0 = 339
    fluxes, p1s, p2s = [], [], []
    for iflux in range(Nfluxes):
        first = s0 + iflux * Npots
        scans = np.arange(first, first + Npots)
        pots, p1, p2 = potential_dependence(scans, show_images=False)
        p1s.append(p1)
        p2s.append(p2)
        fluxes.append(get_flux(scans[0]).mean())
    fig, ax = plt.subplots(figsize=(4,8), ncols=1)
    for i in range(Nfluxes):
        ax.plot(pots, np.array(p1s[i]) - i * 3, 'x-')
        ax.text(max(pots), -i*3, '%.1e/s'%fluxes[i], ha='right')

if 0:
    # look at the diffuse stuff between 111 and 200
    # something builds up as you apply cathodic potential, then almost
    # recovers when you go positive again.
    # what is it?
    fig, ax = plt.subplots(ncols=2, figsize=(8,4))
    scans = np.arange(339, 564)
    pots, diffuse = [], []
    for i, scan in enumerate(scans):
        avg = load_avg(scan)
        pots.append(get_potential(scan))
        r, I = radial_integral(avg)
        diffuse.append(np.sum(I[230:280]))
    fluxes = np.array([get_flux(s).mean() for s in scans])
    diffuse = np.array(diffuse)
    ax[0].plot(pots[:15], diffuse.reshape((15,15))/fluxes.reshape((15,15)))
    ax[1].plot(scans, diffuse.flatten()/fluxes, 'x-')

"""

* Looks like 323-337 is a good baseline at 1e10 ph/s. There a
reversible potential dependence. Also look at detail in the average images.
And dynamics?

* 339-564 diminishing flux

* make sure the high-potential rings agree with in-air data (taking account of
the detector move between scans 247 and 248)

"""
