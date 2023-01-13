"""
Explores the following kinetic model, under steady production of primary
products H, OH, H2, and H2O2 at constant ratio in spurs. The primary
product formation is function of flux density.

(1)  H2 + .OH  -> .H + H2O
(2)  .H + H2O2 -> H2O + .OH
(3)  .OH + .H  -> H2O
(4)  .H + .H   -> H2
(5)  .OH + .OH -> H2O2

Rate constants in 1 / M / s:
(1) k1 = 5e7   Thomas1966 - https://doi.org/10.1021/j100879a503
(2) k2 = 3.6e7 Mezyk1995 - https://doi.org/10.1039/FT9959103127
(3) k3 = 7e9   Thomas1965
(4) k4 = 5e9   Sehested1990 - https://doi.org/10.1016/1359-0197(90)90040-O
(5) k5 = 5e9   Thomas 1965

So 1-2 are slow, and 3-5 are fast and comparable with each other.

In addition, we'll include these O2 reactions scavenger reactions,
(6) .H + O2 -> HO2
(7) 2 HO2 -> H2O2 + O2
(8) HO2 + OH -> O2 + H2O

These are much slower, but catalytic. From Spinks&Wood, in 1 / M / s:
(6) 2.1e4
(7) 8.3e-1
(8) 6e3

We'll do this as coupled ODE:s, based on a species vector:

    [OH, H, H2, H2O2, HO2, O2, H2O], where H2O is net change.

"""

from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np
plt.rc('font', size=8)
plt.ion()

def spur_rates(C, t, flux_density=0, mu=8.4e2):
    # beam-induced (spur) kinetics
    # mu is the exp(-mu * length) attenuation coefficient in 1/m,
    # equivalent to the inverse of the 1/e attenuation length.
    # 7.3e5 at 800 eV
    # 8.4e2 at 8.5 keV
    # 45 at 24 keV
    # 32 at 30 keV

    # the number photon absorption rate per volume:
    r = flux_density * mu  # ph / m3
    # the absorbed power per volume:
    p = r * 8500 * 1.602e-19  # J / m3 / s
    # convert to J / L / s
    p *= 1e-3

    # inhomogeneous (spur) kinetics: balanced G values
    dOH = .3e-6 * p  # mol/J * J/L/s = M/s
    dH = .376e-6 * p
    dH2 = .05e-6 * p
    dH2O2 = .088e-6 * p
    dH2O = -dOH - 2 * dH2O2
    dHO2 = dO2 = 0

    # print('H. photoproduction %.1f M/s' % dH)

    if len(C) == 7:
        return np.array((dOH, dH, dH2, dH2O2, dHO2, dO2, dH2O))
    else:
        return np.array((dOH, dH, dH2, dH2O2))

def homo_rates(C, t):
    # homogeneous kinetics

    include_O2 = (len(C) == 7)

    k1 = 5e7
    k2 = 3.6e7
    k3 = 7e9
    k4 = 5e9
    k5 = 5e9

    if include_O2:
        OH, H, H2, H2O2, HO2, O2, H2O = C

        k6 = 2.1e4
        k7 = 8.3e-1
        k8 = 6e3

        r1 = k1 * H2 * OH
        r2 = k2 * H * H2O2
        r3 = k3 * OH * H
        r4 = k2 * H**2
        r5 = k5 * OH**2
        r6 = k6 * H * O2
        r7 = k7 * HO2**2
        r8 = k8 * HO2 * OH

        dOH = -r1 + r2 - r3 - 2 * r5 - r8
        dH = r1 - r2 - r3 - 2 * r4 - r6
        dH2 = -r1 + r4
        dH2O2 = -r2 + r5 + r7
        dHO2 = r6 - 2 * r7 - r8
        dO2 = -r6 + r7 + r8
        dH2O = r1 + r2 + r3 + r8

        return np.array((dOH, dH, dH2, dH2O2, dHO2, dO2, dH2O))

    else:
        OH, H, H2, H2O2 = C

        r1 = k1 * H2 * OH
        r2 = k2 * H * H2O2
        r3 = k3 * OH * H
        r4 = k2 * H**2
        r5 = k5 * OH**2

        dOH = -r1 + r2 - r3 - 2 * r5
        dH = r1 - r2 - r3 - 2 * r4
        dH2 = -r1 + r4
        dH2O2 = -r2 + r5

        return np.array((dOH, dH, dH2, dH2O2))


def rates(C, t, flux_density=0):
    return (spur_rates(C, t, flux_density)
            + homo_rates(C, t))

if __name__ == '__main__':
    c0 = np.zeros(7)
    c0[5] = 250e-6  # air-saturated
    flux = 1e13 / (100e-6)**2  # Balder, MAX IV

    t = np.logspace(-7, 3)
    c = odeint(rates, c0, t, args=(flux,))
    c_ = odeint(rates, np.zeros((4,)), t, args=(flux,))

    fig = plt.figure(figsize=(3.33,2.5))
    fig.subplots_adjust(bottom=.17, left=.18, right=.99, top=.99)

    lines, labels = [], []
    for i, species in enumerate(['OH$^\\bullet$', 'H$^\\bullet$', 'H$_\mathrm{2}$', 'H$_\mathrm{2}$O$_\mathrm{2}$']):
        l1 = plt.plot(t, c[:, i], '-')
        l2 = plt.plot(t, c_[:, i], '--', color=l1[0].get_color(), lw=2)
        lines.append(l1[0])
        labels.append(species)

    plt.xscale('log')
    plt.yscale('log')
    plt.autoscale(False)
    plt.plot(plt.xlim(), (0.8e-3,)*2, '--', color='gray')
    plt.text(1e-7, 950e-6, 'p(H$_\mathrm{2}$) = 1 atm')
    plt.ylabel('$C_i$ (M)')
    plt.xlabel('$t$ (s)')
    plt.ylim(5e-9, 4e-3)
    plt.legend(lines[::-1], labels[::-1], frameon=False)
    plt.savefig('fig_kinetics.pdf')
