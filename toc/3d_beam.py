"""
Make a pdb with molecules formed randomly in a gaussian focused beam.
The dimensions are schematic and meant for a TOC graphic.
"""

import numpy as np
import matplotlib.pyplot as plt
plt.ion()

# generate a random beam-shaped cloud of points
N = 10000
L = 1000
W0 = 10
W1 = 100
rOH = 1.


def pdb_line(iatom, ires, xyz, aname='H', element='H', altloc='',
             resname='WAT', chainid='A', icode='', occu=1, Bfactor=0,
             charge=0):
    x, y, z = xyz
    return (
        'ATOM  %5u %4s%1s%3s %1s%4u%1s   %8.3f%8.3f%8.3f'
        '%6.2f%6.2f           % 2s%2s\n' % (
            iatom, aname, altloc, resname, chainid, ires, icode,
            x, y, z, occu, Bfactor, element, charge))


# make a list of random base coordinates
z = np.random.random(size=N).reshape((-1, 1)) * L  # z axis
xy = np.random.normal(size=(N, 2))  # sigma = 1
xy *= (np.abs(z - L / 2) + W0) / (L / 2) * W1 / 2
coords = np.concatenate((xy, z), axis=1)

# make some of these H, others OH, and write to file
fp = open('3d_beam.pdb', 'w')
iatom = 0
for imol in range(N):
    if np.random.rand() < 0.25:
        # OH - make a little H moon in a random direction
        ok = False
        while not ok:
            x = np.random.rand(3) * 2 - 1  # [-1, 1]
            if np.linalg.norm(x) < 1:
                ok = True
            x = x / np.linalg.norm(x) * rOH
            Hpos = coords[imol] + x
        fp.write(pdb_line(iatom, imol, coords[imol], aname='O', element='O'))
        fp.write(pdb_line(iatom + 1, imol, Hpos, aname='H', element='H'))
        iatom += 2
    else:
        # H
        fp.write(pdb_line(iatom, imol, coords[imol], aname='H', element='H'))
        iatom += 1
fp.close()
