"""
Assembles data with physical units on the autocorrelaction constraint.
"""

import numpy as np
import matplotlib.pyplot as plt
plt.ion()

from bcdiass.utils import C, M
from bcdiass.utils import generate_initial, generate_envelope, pre_align_rolls
from bcdiass.utils import ProgressPlot

# input and parameters
data = np.load('picked.npz')['data']
center = np.load('picked.npz')['center']
Nj, Nl, ml = 20, 20, 1
shape = 64
fudge = 5e-4
fudge_max = 1e-2
increase_every = 5
CENTER = [-1700,256] # relative to whole detector origin
CENTER[0] -= (center[0] - data.shape[-1]//2) # relative to picked frames' origin
CENTER[1] -= (center[1] - data.shape[-1]//2)

# physics
theta = 19 # (111) at 2theta=37.9
psize = 55e-6
distance = .120
hc = 4.136e-15 * 3.000e8
E = 8500.
Q12 = psize * 2 * np.pi / distance / (hc / E) * data.shape[-1]
dq12 = Q12 / data.shape[-1]
dq3 = dq12 * 2  # rough q-space aspect ratio
Q3 = dq3 * Nj
Dmax = 45e-9

# pre-roll and crop
data, rolls = pre_align_rolls(data, roll_center=CENTER, plot=True, threshold=3)
crop = (data.shape[-1] - shape) // 2
data = data[:, crop:-crop, crop:-crop]

# now assemble!
envelope = generate_envelope(Nj, data.shape[-1], Q=(Q3, Q12, Q12), Dmax=Dmax, theta=theta)
W = generate_initial(data, Nj)
p = ProgressPlot()
for i in range(100):
    print(i)
    W, error = C(W, envelope)
    W, Pjlk, timing = M(W, data, Nl=Nl, ml=ml, beta=fudge,
                        nproc=24,
                        force_continuity=5,
                        roll_center=CENTER,
                        center_model=True)
    [print(k, '%.3f'%v) for k, v in timing.items()]
    p.update(W, Pjlk, [], vmax=1, log=True)

    if i and (fudge < fudge_max) and (i % increase_every) == 0:
        fudge *= 2**(1/2)
        print('increased fudge to %e'%(fudge))

np.savez('assembled.npz', W=W, Pjlk=Pjlk, rolls=rolls, Q=(Q3, Q12, Q12))
