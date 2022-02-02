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
(1) k1 = 5e7   (https://doi.org/10.1021/j100879a503)
(2) k2 = 3.6e7 (https://doi.org/10.1039/FT9959103127)
(3) k3 = 7e9   (JK Thomas, Trans. Faraday Soc. 61, 1965, 702)
(4) k4 = 5e9   (https://doi.org/10.1016/1359-0197(90)90040-O)
(5) k5 = 5e9   (Thomas 1965)

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
plt.rc('font', size=14)
plt.ion()


def spur_rates(C, t, flux_density=0, beam_off=None):
    # beam-induced (spur) kinetics

    # caluclate spur rates for 8.5 keV
    mu = 8.4e2  # 1/m
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

    # make sure masses balance
    # assert np.allclose(dOH + 2*dH2O2 + 2*dHO2 + 2*dO2 + dH2O, 0)  # oxygen
    # assert np.allclose(dOH + dH + 2*dH2 + 2*dH2O2 + dHO2 + 2*dH2O, 0)  # hydrogen

    # optionally switch off the beam at some time
    if beam_off and (t > beam_off):
        dOH = dH = dH2 = dH2O2 = 0

    return np.array((dOH, dH, dH2, dH2O2, dHO2, dO2, dH2O))

def homo_rates(C, t):
    # homogeneous kinetics

    # unpack
    OH, H, H2, H2O2, HO2, O2, H2O = C

    k1 = 5e7
    k2 = 3.6e7
    k3 = 7e9
    k4 = 5e9
    k5 = 5e9
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

    # make sure masses balance
    # assert np.allclose(dOH + 2*dH2O2 + 2*dHO2 + 2*dO2 + dH2O, 0)  # oxygen
    # assert np.allclose(dOH + dH + 2*dH2 + 2*dH2O2 + dHO2 + 2*dH2O, 0)  # hydrogen

    return np.array((dOH, dH, dH2, dH2O2, dHO2, dO2, dH2O))

def rates(C, t, flux_density=0, beam_off=None):
    return (spur_rates(C, t, flux_density, beam_off)
            + homo_rates(C, t))

if __name__ == '__main__':
    c0 = np.zeros(7)
    c0[5] = 250e-6  # air-saturated
    t = np.logspace(-7, 3)
    # flux = 7e10 / (105e-9)**2  # nanomax
    flux = 1e13 / (100e-6)**2  # balder
    # flux = 1e9 / (50e-9)**2  # softimax, 700 eV!
    c = odeint(rates, c0, t, args=(flux, None))
    c_ = odeint(rates, np.zeros_like(c0), t, args=(flux, None))
    plt.figure(figsize=(6,4))
    plt.subplots_adjust(bottom=.2, left=.15, right=.99, top=.99)
    # don't include T, without diffusion it just grows of course.
    cols = 'kcrmgb'
    for i, species in enumerate(['OH', 'H', 'H2', 'H2O2']):  # 'HO2', 'O2'
        plt.plot(t, c[:, i], '-', color=cols[i], label=species)
        plt.plot(t, c_[:, i], ':', color=cols[i])
#    plt.plot(t, c[:, 4], label='HO2')
#    plt.plot(t, c[:, 5], label='O2')
    plt.xscale('log')
    plt.yscale('log')
    plt.autoscale(False)
    plt.plot(plt.xlim(), (780e-6,)*2, '--', color='gray')
    plt.text(1e-7, 900e-6, 'p(H2) = 1 atm')
    plt.ylabel('$C$ / (mol / L)')
    plt.xlabel('$t$ / s')
    plt.ylim(5e-9, 4e-3)
    plt.legend(frameon=False)
    plt.savefig('kinetics.pdf')