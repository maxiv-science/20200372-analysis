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


r0 = 50e-9
r = np.logspace(-9, -3, 500)
t = np.logspace(-7, 1, 200)
y0 = np.zeros(6 * len(r))
y0.reshape((6, -1))[-1, :] = 298  # a view
D = np.ones(6) * 5e-9
D[-1] = .143 * 1e-6  # m2 / s
D[-1] = 0  # turn off heat conduction
flux_density = 7e11 / (100e-9)**2

c = odeint(dcdt, y0, t, args=(D, r, r0, flux_density))
c = c.reshape((len(t), len(D), -1))

edge = np.where(r <= r0)[-1][-1]
mol_per_length = np.sum(
    c[:, :, :edge]
    * 2 * np.pi * r[:edge]
    * np.diff(r)[:edge].reshape((1, -1)),
    axis=-1,
)
av_conc = mol_per_length / (np.pi * r0**2)
plt.figure()
plt.plot(t, av_conc[:, :-2], label=['OH', 'H', 'H2', 'H2O2'])
plt.xscale('log')
#plt.yscale('log')
plt.legend()
