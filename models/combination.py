"""
A combination of radical kinetics and cylindrical diffusion.
"""

from diffusion import laplacian, H
from kinetics import homo_rates, spur_rates

import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
plt.rc('font', size=8)
plt.ion()
plt.close('all')


def dcdt(C, t, D, r, r0, flux_density=0, mu=8.4e2):

    # odeint does 1d vectors, so our species are stacked - work on a view
    n_species = len(D)
    C_rs = C.reshape((n_species, -1))
    dC = np.zeros_like(C_rs)

    # call the diffusion term on each species
    for i in range(n_species):
        dC[i, :] += D[i] * laplacian(C_rs[i], r, dim=2)

    # spur rates are (n_species,)
    dC[:, :] += (
        spur_rates(C_rs, t, flux_density, mu).reshape((-1, 1)) * H(r0 - r)
    )

    # homogeneous reactions
    dC[:, :] += homo_rates(C_rs, t)

    return dC.flatten()

if 0:
    mus = [7.3e5, 8.4e2, 45]
    r0 = 50e-9
    r = np.logspace(-9, -3, 500)
    t = np.logspace(-9, 0, 200)
    y0 = np.zeros(4 * len(r))
    # [OH, H, H2, H2O2]
    D = np.array((2.3, 5, 5, 2)) * 1e-9  # m2/s
    flux_density = 7e10 / (3.14 * r0**2)

    av_concs = []
    for j, mu in enumerate(mus):
        print(int(round(mu)))
        c = odeint(dcdt, np.zeros_like(y0), t, args=(D, r, r0, flux_density, mu))
        c = c.reshape((len(t), len(D), -1))

        edge = np.where(r <= r0)[-1][-1]
        mol_per_length = np.sum(c[:, :, :edge] * 2 * np.pi * r[:edge] * np.diff(r)[:edge].reshape((1, -1)), axis=-1)
        av_concs.append(mol_per_length / (np.pi * r0**2))
    av_concs=np.array(av_concs)
    np.savez('combination.npz',
             av_concs=av_concs, t=t, mus=np.array(mus))
else:
    dct = np.load('combination.npz')
    av_concs = dct['av_concs']
    mus = dct['mus']
    t = dct['t']

# 8.5 keV for the paper
av_conc = av_concs[1]
fig = plt.figure(figsize=(3.33,2.5))
fig.subplots_adjust(bottom=.17, left=.14, right=.99, top=.98)
lines, labels = [], []
for j, lbl in enumerate(['OH$^\\bullet$', 'H$^\\bullet$', 'H$_\mathrm{2}$', 'H$_\mathrm{2}$O$_\mathrm{2}$']):
    lines.append(plt.plot(t, av_conc[:, j] * 1e3, label=lbl)[0])
    labels.append(lbl)
plt.xscale('log')
plt.xlim([1e-8, 2])
plt.ylabel('C$_i$  (mM)')
plt.xlabel('t  (s)')
plt.legend(lines[::-1], labels[::-1], frameon=False)
plt.savefig('fig_combination.pdf')

# Other energies for the SI
fig, ax = plt.subplots(figsize=(6,2.5), ncols=2)
fig.subplots_adjust(bottom=.17, left=.1, right=.99, top=.98)

# 24 keV
ax[0].plot(t, av_concs[2, :, 0] * 1e3, label='OH$^\\bullet$')
ax[0].plot(t, av_concs[2, :, 1] * 1e3, label='H$^\\bullet$')
ax[0].plot(t, av_concs[2, :, 2] * 1e3, label='H$_\mathrm{2}$')
ax[0].plot(t, av_concs[2, :, 3] * 1e3, label='H$_\mathrm{2}$O$_\mathrm{2}$')
handles, labels = ax[0].get_legend_handles_labels()
ax[0].legend(handles[::-1], labels[::-1], frameon=False)
ax[0].set_xscale('log')
ax[0].set_ylabel('C$_i$  (mM)')
ax[0].set_xlabel('t  (s)')

# 800 eV
ax[1].plot(t, av_concs[0, :, 0] * 1e3, label='OH$^\\bullet$')
ax[1].plot(t, av_concs[0, :, 1] * 1e3, label='H$^\\bullet$')
ax[1].plot(t, av_concs[0, :, 2] * 1e3 * .1, label='H$_\mathrm{2}$ (x0.1)')
ax[1].plot(t, av_concs[0, :, 3] * 1e3 * .1, label='H$_\mathrm{2}$O$_\mathrm{2}$ (x0.1)')
handles, labels = ax[1].get_legend_handles_labels()
ax[1].legend(handles[::-1], labels[::-1], frameon=False)
ax[1].set_xscale('log')
ax[1].set_xlabel('t  (s)')
ax[1].text(0, 0, 'E=')

fig.text(.005, .99, 'a)', va='top')
fig.text(.525, .99, 'b)', va='top')

plt.savefig('fig_combination_SI.pdf')
