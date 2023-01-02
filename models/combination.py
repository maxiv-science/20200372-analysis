"""
A combination of radical kinetics and cylindrical diffusion.
"""

from diffusion import laplacian, H
from kinetics import homo_rates, spur_rates

import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
plt.rc('font', size=14)
plt.ion()


def dcdt(C, t, D, r, r0, flux_density=0):

    # odeint does 1d vectors, so our species are stacked - work on a view
    n_species = len(D)
    C_rs = C.reshape((n_species, -1))
    dC = np.zeros_like(C_rs)

    # call the diffusion term on each species
    for i in range(n_species):
        dC[i, :] += D[i] * laplacian(C_rs[i], r, dim=2)

    # spur rates are (n_species,)
    dC[:, :] += (
        spur_rates(C_rs, t, flux_density).reshape((-1, 1)) * H(r0 - r)
    )

    # homogeneous reactions
    dC[:, :] += homo_rates(C_rs, t)

    return dC.flatten()

if True:
    r0 = 50e-9
    r = np.logspace(-9, -3, 500)
    t = np.logspace(-6, 1, 200)
    y0 = np.zeros(7 * len(r))
    y0.reshape((7, -1))[5, :] = 250e-6  # air-saturated
    D = np.ones(7) * 5e-9
    flux_density = 7e10 / (3.14 * r0**2)

    print('with O2...')
    c = odeint(dcdt, y0, t, args=(D, r, r0, flux_density))
    c = c.reshape((len(t), len(D), -1))
    print('without O2...')
    c_ = odeint(dcdt, np.zeros_like(y0), t, args=(D, r, r0, flux_density))
    c_ = c_.reshape((len(t), len(D), -1))

    edge = np.where(r <= r0)[-1][-1]
    mol_per_length = np.sum(c[:, :, :edge] * 2 * np.pi * r[:edge] * np.diff(r)[:edge].reshape((1, -1)), axis=-1)
    mol_per_length_ = np.sum(c_[:, :, :edge] * 2 * np.pi * r[:edge] * np.diff(r)[:edge].reshape((1, -1)), axis=-1)

    av_conc = mol_per_length / (np.pi * r0**2)
    av_conc_ = mol_per_length_ / (np.pi * r0**2)
    np.savez('combination.npz', av_conc=av_conc, av_conc_=av_conc_, t=t)
else:
    dct = np.load('combination.npz')
    av_conc = dct['av_conc']
    av_conc_ = dct['av_conc_']
    t = dct['t']

plt.figure()
plt.subplots_adjust(bottom=.2, left=.15, right=.99, top=.99)
cols = 'kcrmgb'
for i, lbl in enumerate(['OH', 'H', 'H2', 'H2O2']):
    plt.plot(t, av_conc[:, i] * 1e3, color=cols[i], label=lbl)
    plt.plot(t, av_conc_[:, i] * 1e3, ':', color=cols[i])
plt.xscale('log')
plt.ylabel('C / ($10^{-3}$ mol / L)')
plt.xlabel('t / s')
plt.legend(frameon=False)
plt.savefig('combination.pdf')
