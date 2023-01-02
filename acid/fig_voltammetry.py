import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from utils import (load_avg, get_flux, get_potential, mask, PATH,
                   radial_integral)

matplotlib.rcParams.update({'font.size': 8})
plt.ion()

fig, ax = plt.subplots(figsize=(3.33,2))
plt.subplots_adjust(left=.17, right=.99, bottom=.2, top=.99)

def load_cv(nr, n_bin=100):
    path = '/data/visitors/nanomax/20200372/2021062308/process/cv/cv%d.npz'
    dct = np.load(path % nr)
    E, I = dct['E'], dct['I'] * 1e6
    N = E.shape[0] // n_bin * n_bin
    E = E[:N].reshape(-1, n_bin).sum(axis=1) / n_bin
    I = I[:N].reshape(-1, n_bin).sum(axis=1) / n_bin
    return E, I

E1, I1 = load_cv(22, n_bin=500)
E2, I2 = load_cv(23, n_bin=500)

ax.plot(E1, I1)
ax.plot(E2, I2)
ax.set_xlabel('$E$ (V vs Ag/AgCl, sat.)')
ax.set_ylabel('$I$ ($\mu$A)')
ax.text(-.19, -45, 'H$_{\mathrm{abs}}$')
ax.text(-.26, -65, 'H$_{\mathrm{2}}$(g)')
ax.set_ylim((-70, 42))
ax.set_xlim((-.27, .015))
plt.savefig('fig_voltammetry.pdf')