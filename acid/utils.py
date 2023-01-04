import h5py
import numpy as np
from nmutils.utils.ion_chamber import Ionchamber
import os

PATH = '/data/visitors/nanomax/20200372/2021062308/raw/sample/%06u.h5'
mask = np.load('mask.npy')
CENTER = -1500, 515//2


def load_avg(scannr):
    npy = 'npy/%06u.npy' % scannr
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

def radial_integral(image, center=CENTER):
    tops = np.arange(10, 500) # tops of the rings on the detector
    r = tops - center[0]
    ii, jj = np.indices(image.shape)
    rr = np.sqrt((ii-center[0])**2 + (jj-center[1])**2)
    avgs = []
    for r_ in r:
        mask = (np.abs(r_ - rr) <= .5)
        avgs.append(np.mean(image[mask]))
    return tops, np.array(avgs)
