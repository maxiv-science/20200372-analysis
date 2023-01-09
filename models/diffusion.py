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
plt.rc('font', size=8)
plt.ion()


# global parameters
r0 = 50e-9  # 100 nm beam
D = 5e-5 * 1e-4   # 5e-5 cm2/s for H2, converted to m2/s
c0 = 0   # initial solute conc in beam, M
v = .0005    # M/s, rate of solute generation
r = np.logspace(-10, -3, 1000)
t = np.logspace(-8, 1, 200)
dimensions = 2  # dimensions (1=flat, 2=cylindrical, 3=spherical)


def laplacian(a, r, dim=2):
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


def Cmean(C, r, r0):
    """
    Average concentration across a beam of radius r0, as function of
    time. Tricky because of the uneven r spacing.

    C = C[t, r]
    """
    edge = np.where(r <= r0)[-1][-1]
    av = (
        np.sum(
            C[:, :edge] * 2 * r[:edge]
            * np.diff(r)[:edge].reshape((1, -1)),
            axis=1)) / r0**2
    return av

if __name__ == '__main__':
    def dydt(y, t, r, v, D, r0):
        stp = r[1] - r[0]
        return D * laplacian(y, r, dim=dimensions) + v * H(r0 - r)

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

    if 1:
        fig, ax = plt.subplots(ncols=2, figsize=(6,2.5),
            gridspec_kw=dict(width_ratios=[2, 1]))
        plt.subplots_adjust(bottom=.2, left=.07, right=.98, top=.95, wspace=.3)

        # a) general solution
        dct = np.load('diffusion.npz')
        for k in dct.keys():
            exec("%s = dct['%s']" % (k, k))
        for v, D, r0, c in zip(vs, Ds, r0s, conc):
          av_conc = Cmean(c, r, r0)
          ax[0].plot(
              np.log10(2 * t * D / r0**2),
              av_conc / (v * r0**2 / D / np.sqrt(3)),
              lw=.5, color='k')

        ax[0].xaxis.set_ticks(range(-6, 9, 2))
        ax[0].yaxis.set_ticks(range(0, 9, 2))
        ax[0].set_xlabel("$\log(\\bar{t}) = \log(2 D t / {r_0}^2)$")
        ax[0].set_ylabel("$\\bar{c} = |c| \cdot D \sqrt{3} / (v * {r_0}^2)$")
        ax[0].set_aspect('equal')
        mark, xmin, ymin = 8, ax[0].get_xlim()[0], ax[0].get_ylim()[0]
        plt.autoscale(False)
        ax[0].plot([mark, mark], [ymin, mark], color='gray', linestyle=':')
        ax[0].plot([xmin, mark], [mark, mark], color='gray', linestyle=':')

        # b) specific NanoMAX solution
        # Cbar = |C|_beam * D * sqrt(3) / (v * r_0**2))
        # tbar = t * 2 * D /  (r_0**2)
        # Cbar = log10(tbar)
        #
        # Now, at 8.5 keV and 7e10/s on the sample, H2 is produced
        # at v=1505 M/s.
        t = np.logspace(-7, 0)
        v = 1505 # M/s converted to mol/m3/s
        r0 = 50e-9 # m
        D = 5e-5 * 1e-4  # m2 / s
        Cmean = v * r0**2 / (D * np.sqrt(3)) * np.log10(t * 2 * D / r0**2)
        Cmean[0] = 0
        ax[1].plot(t, Cmean * 1e3, 'k')
        #ax[1].set_xscale('log')
        ax[1].set_xlim(-.01, 1)
        ax[1].set_ylim(0, 3)
        ax[1].set_ylabel('$|c|$ (mM)')
        ax[1].set_xlabel('$t$ (s)')
        ax[1].set_yticks([0, 1, 2, 3])

        fig.text(.01, .99, 'a)', va='top')
        fig.text(.65, .99, 'b)', va='top')
        plt.savefig('fig_diffusion.pdf')
