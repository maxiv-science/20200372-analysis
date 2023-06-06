"""
Try finding out what the net effect of focusing is, you concentrate
the flux but you also make diffusion more efficient.

Not currently used in the manuscript.

"""

from diffusion import laplacian, H
from kinetics import homo_rates, spur_rates, rates

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
    r0s = np.array((50e-9, 500e-9, 2.5e-6, 25e-6, 250e-6))
    r = np.logspace(-9, -3, 500)
    t = np.logspace(-9, 0, 200)
    y0 = np.zeros(4 * len(r))
    # [OH, H, H2, H2O2]
    D = np.array((2.3, 5, 5, 2)) * 1e-9  # m2/s

    av_concs, concs = [], []
    for r0 in r0s:
        print(r0)
        flux_density = 7e10 / (3.14 * r0**2)

        c = odeint(dcdt, np.zeros_like(y0), t, args=(D, r, r0, flux_density))
        c = c.reshape((len(t), len(D), -1))

        edge = np.where(r <= r0)[-1][-1]
        mol_per_length = np.sum(c[:, :, :edge] * 2 * np.pi * r[:edge] * np.diff(r)[:edge].reshape((1, -1)), axis=-1)
        concs.append(c)
        av_concs.append(mol_per_length / (np.pi * r0**2))

    # pure kinetics of the last one
    c_ = odeint(rates, np.zeros((4,)), t, args=(flux_density,))
    av_concs.append(c_)

    np.savez('focusing.npz',
             av_concs=np.array(av_concs), t=t, r=r, r0s=r0s)
else:
    dct = np.load('focusing.npz')
    av_concs = dct['av_concs']
    t = dct['t']
    r0s = dct['r0s']

# plot
fig, ax = plt.subplots(figsize=(6,3), ncols=3, nrows=2, sharex=True)
fig.subplots_adjust(bottom=.14, left=.08, right=.99, top=.98, hspace=.1)
ax = ax.flatten()
for i, av_conc in enumerate(av_concs):
    lines, labels = [], []
    for j, lbl in enumerate(['OH$^\\bullet$', 'H$^\\bullet$', 'H$_\mathrm{2}$', 'H$_\mathrm{2}$O$_\mathrm{2}$']):
        unit = (1e3 if i < 3 else 1e6)
        lines.append(ax[i].plot(t, av_conc[:, j] * unit, label=lbl)[0])
        labels.append(lbl)
    ax[i].set_xscale('log')
    ax[i].tick_params(axis='y', pad=1)
    if i < 5:
        d_nm = r0s[i] * 2 * 1e9
        if (d_nm // 1000 > 0):
            val = d_nm / 1000
            unit = '$\mu$m'
        else:
            val = d_nm
            unit = 'nm'
        ax[i].text(1e-9, ax[i].get_ylim()[1] * .95,
                   '$d=%d$ %s' % (val, unit), va='top')

ax[4].set_xlabel('t  (s)')
ax[0].set_ylabel('C$_i$  (mM)')
ax[3].set_ylabel('C$_i$  ($\mu$M)')
ax[-1].legend(lines[::-1], labels[::-1], frameon=False)
plt.savefig('fig_focusing.pdf')
