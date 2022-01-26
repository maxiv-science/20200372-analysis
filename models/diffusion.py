"""
Numerically explore the diffusion problem where a species is constantly
generated within a range (r < r0) at rate v, in various geometries.

In the cylindrical geometry, which is a model for photo-generated
species inside a beam, the average concentration for (r < r0) does not
settle at a steady state, but grows like

 |c| = (sqrt(3) * v * r0**2 / D) * log(t * D / r0**2),

where the log argument is unitless and the first factor has units of
concentration. For constant beam flux and assuming a linear
photoreduction, the number of photoreduced molecules is constant across
the beam, and the number v therefore goes as 1/r0**2, which means that
the only beam size dependence is in the logarithm.

Anyway, accumulation starts at t * D / r0**2 = 1, that is when
sqrt(t*D) = r0, which seems very reasonable.

"""


import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
plt.rc('font', size=14)
plt.ion()


# global parameters
r0 = 50e-9  # 100 nm beam
D = 5e-5 * 1e-4   # 5e-5 cm2/s for H2, converted to m2/s
c0 = 0   # initial solute conc in beam, M
v = .0005    # M/s, rate of solute generation
r = np.logspace(-9, -3, 500)
t = np.logspace(-8, 1, 200)
dimensions = 2  # dimensions (1=flat, 2=cylindrical, 3=spherical)


def laplacian(a, dim=2):
    """
    In flat, cylindrical, or spherical coordinates.
    """

    # using finite differences on an uneven grid, from this paper:
    # https://doi.org/10.3402/tellusa.v22i1.10155
    # spacing from an element to its right:
    hi = np.diff(r, append=(2*r[-1]-r[-2]))
    # spacing from an element to its left:
    hi_minus_1 = np.roll(hi, 1)
    # eq 2.2, fprime half way to its neighbors:
    fprime_i_minus_half = (a - np.roll(a, 1)) / hi_minus_1
    fprime_i_plus_half = (np.roll(a, -1) - a) / hi
    # eq 2.3, first equality, weighted average:
    fprime = (hi_minus_1 / (hi + hi_minus_1) * fprime_i_plus_half
              + hi / (hi + hi_minus_1) * fprime_i_minus_half)
    # put that together with edge values
    first = np.zeros_like(a)
    first[1:-1] = fprime[1:-1]
    first[0] = 0  # boundary condition
    first[-1] = 0
    # eq 2.4 for the second derivative
    second = 2 * (fprime_i_plus_half - fprime_i_minus_half) / (hi + hi_minus_1)
    second[0] = (first[1] - first[0]) / hi[0]
    second[-1] = 0

    res = np.zeros_like(a)
    res[1:] = second[1:] + (dim - 1) / r[1:] * first[1:]
    res[0] = second[0]  # probably true as first derivative=0

    return res


def H(x):
    return (x >= 0).astype(int)


def dydt(y, t, r, v, D, r0):
    stp = r[1] - r[0]
    return D * laplacian(y, dim=dimensions) + v * H(r0 - r)


if 0:
    # single simulation of H2 generation at NanoMAX in real units: 3 mM
    v = 1505  # M / s
    r0 = 50e-9
    D = 5e-9
    y0 = np.zeros_like(r)
    c = odeint(dydt, y0, t, args=(r, v, D, r0))

    edge = np.where(r <= r0)[-1][-1]
    mol_per_length = np.sum(
        c[:, :edge]
        * 2 * np.pi * r[:edge]
        * np.diff(r)[:edge].reshape((1, -1)),
        axis=1,
    )
    av_conc = mol_per_length / (np.pi * r0**2)
    plt.plot(t, av_conc)


if 0:
    # simulate across a relevant parameter space and save the data.
    vs = np.logspace(0, 3, 4)     # M / s - estimated from G values
    vs *= 1e3                     # mol / m3 / s
    Ds = np.logspace(-7, -4, 4)   # cm2 / s
    Ds *= 1e-4                    # m2 / s
    r0s = np.logspace(1, 3, 3)    # nm
    r0s *= 1e-9                   # m
    vs, Ds, r0s = np.reshape(np.meshgrid(vs, Ds, r0s), (3, -1))
    conc = []
    for v, D, r0 in zip(vs, Ds, r0s):
        print(v, D, r0)
        y0 = (r < r0).astype(int) * c0
        conc.append(odeint(dydt, y0, t, args=(r, v, D, r0)))
    np.savez('diffusion.npz', conc=np.array(conc), r=r, t=t, vs=vs, Ds=Ds, r0s=r0s)



if 0:
    fig, ax = plt.subplots(ncols=1, figsize=(6,4))
    plt.subplots_adjust(bottom=.2, left=.15, right=.99, top=.99)

    dct = np.load('diffusion.npz')
    for k in dct.keys():
        exec("%s = dct['%s']" % (k, k))
    for v, D, r0, c in zip(vs, Ds, r0s, conc):
        edge = np.where(r <= r0)[-1][-1]
        mol_per_length = np.sum(
            c[:, :edge]
            * 2 * np.pi * r[:edge]
            * np.diff(r)[:edge].reshape((1, -1)),
            axis=1,
        )
        av_conc = mol_per_length / (np.pi * r0**2)
        ax.plot(
            np.log10(t * D / r0**2),
            av_conc / (v * r0**2 / D / np.sqrt(3)),
            lw=1,)
#    x = np.linspace(-.5, 10)
#    plt.plot(x, x, 'k--')
    ax.xaxis.set_ticks(range(-6, 9, 2))
    ax.yaxis.set_ticks(range(0, 9, 2))
    ax.set_xlabel("$log(\\bar{t}) = log(t * D / {r_0}^2)$")
    ax.set_ylabel("$\\bar{c} = c / (v * {r_0}^2 / [D \\sqrt{3}])$")
    ax.set_aspect('equal')
    mark, xmin, ymin = 8, ax.get_xlim()[0], ax.get_ylim()[0]
    plt.autoscale(False)
    ax.plot([mark, mark], [ymin, mark], color='gray', linestyle=':')
    ax.plot([xmin, mark], [mark, mark], color='gray', linestyle=':')
    plt.savefig('diffusion.pdf')


# ############# ok so real units?
# For water, 1 um tranmits .99916 according to CXRO.
# We have T = I/I0 = exp(-mu d), so mu is around 8.4 cm^-1. NIST says
# 10.4 cm^-1 at 8 keV assuming density 1 g/cm3, so adds up.
#
# This means that for a small path length d, we dump
# I0 - I / V = I0 (1 - exp[-mu d]) / (r0**2 * d) per volume,
# take the low d limit and it's (I0 mu) / r0**2.
#
# Choppin et al say that G for H2 is 0.047 umol/J. So that would mean...
r0 = 50e-9  # m
I0 = 7e10   # photons per second
mu = 8.4e2  # 1/m
v = (I0 * mu / r0**2     # absorbed photons per m3 per second
     * 8500 * 1.602e-19  # photon energy in J
     * .047 * 1e-6       # mol created per J
     * 1e-3)             # conversion to mol/L/s.

# now do the calculation for H2
D = 5e-5 * 1e-4  # m2 / s
C_H2 = v * r0**2 / (np.sqrt(3) * D) * np.log10(1 * D / r0**2)

# for e(aq)
D = 4.5e-5 * 1e-4  # m2/s
v = v / .047 * .28
C_e = v * r0**2 / (np.sqrt(3) * D) * np.log10(1 * D / r0**2)

# for OH
D = 2e-9  # m2/s
C_OH = v * r0**2 / (np.sqrt(3) * D) * np.log10(1 * D / r0**2)

# actually e- (aq) has roughly the same D as H2. So just scales with v.
