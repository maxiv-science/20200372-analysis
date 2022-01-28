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

We'll do this as coupled ODE:s, based on a species vector:

    [OH, H, H2, H2O2, H2O, T], where H2O is net change and T is temp.

"""

from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np
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

    # heating
    p_cm3 = p * 1e-3  # J / cm3 / s
    heat_cap = 4.18  # J / g / K
    rho = 0.997  # g/cm3
    dT = p_cm3 / (heat_cap * rho)  # K / s

    # inhomogeneous (spur) kinetics: balanced G values
    dOH = .3e-6 * p  # mol/J * J/L/s = M/s
    dH = .376e-6 * p
    dH2 = .05e-6 * p
    dH2O2 = .088e-6 * p
    dH2O = -dOH - 2 * dH2O2

    # make sure masses balance
    assert np.allclose(dOH + 2 * dH2O2 + dH2O, 0)
    assert np.allclose(dOH + dH + 2 * (dH2 + dH2O2 + dH2O), 0)

    # optionally switch off the beam at some time
    if beam_off and (t > beam_off):
        dOH = dH = dH2 = dH2O2 = 0

    return np.array((dOH, dH, dH2, dH2O2, dH2O, dT))

def homo_rates(C, t):
    # homogeneous kinetics

    # unpack
    OH, H, H2, H2O2, H2O, T = C

    k1 = 5e7
    k2 = 3.6e7
    k3 = 7e9
    k4 = 5e9
    k5 = 5e9

    r1 = k1 * H2 * OH
    r2 = k2 * H * H2O2
    r3 = k3 * OH * H
    r4 = k2 * H**2
    r5 = k5 * OH**2

    dOH = -r1 + r2 - r3 - 2 * r5
    dH = r1 - r2 - r3 - 2 * r4
    dH2 = -r1 + r4
    dH2O2 = -r2 + r5
    dH2O = r1 + r2 + r3
    dT = np.zeros_like(dOH, dtype=float)

    # make sure masses balance
    assert np.allclose(dOH + 2 * dH2O2 + dH2O, 0)
    assert np.allclose(dOH + dH + 2 * (dH2 + dH2O2 + dH2O), 0)

    return np.array((dOH, dH, dH2, dH2O2, dH2O, dT))

def rates(C, t, flux_density=0, beam_off=None):
    return (spur_rates(C, t, flux_density, beam_off)
            + homo_rates(C, t))

if __name__ == '__main__':
    c0 = np.zeros(6)
    c0[-1] = 298
    t = np.logspace(-7, 3)
    nanomax = 7e10 / (105e-9)**2
    balder = 1e12 / (100e-6)**2
    softimax = 1e9 / (50e-9)**2  # 700 eV!
    c = odeint(rates, c0, t, args=(balder, None))
    plt.figure()
    # don't include T, without diffusion it just grows of course.
    plt.plot(t, c[:, :-1], label=['OH', 'H', 'H2', 'H2O2', 'H2O'])
    plt.xscale('log')
    plt.yscale('log')
    plt.autoscale(False)
    plt.plot(plt.xlim(), (780e-6,)*2, '--', color='gray', label='p(H2) = 1 atm')
    plt.legend()
