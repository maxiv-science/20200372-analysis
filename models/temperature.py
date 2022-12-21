"""
Numerically explore the diffusion problem where a cylindrical beam
heats up an aqueous system.
"""


import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from diffusion import laplacian, H, Cmean
plt.rc('font', size=14)
plt.ion()


def dydt(y, t, r, v, D, r0):
    return D * laplacian(y, r, dim=2) + v * H(r0 - r)


# parameters
r0 = 50e-9
flux_density = 7e10 / np.pi / r0**2
# r0 = 50e-6
# flux_density = 1e13 / np.pi / r0**2
D = 0.143 * 1e-6  # mm2/s converted to m2/s

# work out the heating rate first
mu = 840  # 1/m
# the photon absorption rate per volume:
r = flux_density * mu  # ph / m3
# the absorbed power per volume:
p = r * 8500 * 1.602e-19  # J / m3 / s
# convert to J / cm3 / s
p_cm3 = p * 1e-6
heat_cap = 4.18  # J / g / K
rho = 0.997  # g/cm3
dT = p_cm3 / (heat_cap * rho)  # K / s

# single simulation of heating at NanoMAX in real units:
r = np.logspace(-9, -3, 200)
t = np.logspace(-6, 0, 200)
T0 = np.ones_like(r) * 298
T = odeint(dydt, T0, t, args=(r, dT, D, r0))

fig, ax = plt.subplots(ncols=2, constrained_layout=True, figsize=(8, 3))
ax[0].plot(t, T[:, 0] - T0[0])
ax[0].plot(t, dT * r0**2 / (D * np.sqrt(3)) * np.log10(2 * t * D / r0**2), '--')
for i in range(0, len(t), 10):
    ax[1].plot(r, T[i, :] - T0[0])
ax[1].set_xscale('log')
ax[0].set_xscale('log')
ax[0].set_xlabel('$t$ / s')
ax[1].set_xlabel('$r$ / m')
ax[0].set_ylabel('$\Delta T(r=0)$ / K')
ax[1].set_ylabel('$\Delta T$ / K')
ax[1].axvline(r0, linestyle='--', color='k')
