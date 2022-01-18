import numpy as np
import matplotlib.pyplot as plt

dct = np.load('assembled.npz')
W = dct['W']
Pjlk = dct['Pjlk']

fig, ax = plt.subplots(ncols=4, figsize=(12,3))

Nj = Pjlk.shape[0]
[a.clear() for a in ax]
ax[0].imshow(np.log10(W.sum(1)))
ax[1].imshow(np.log10(W.sum(0)))
Pjk = np.sum(Pjlk, axis=1)
ax[2].imshow(np.abs(Pjk), vmax=np.abs(Pjk).max()/2, aspect='auto')
Plk = np.sum(Pjlk, axis=0)
if Plk.shape[0] > 1:
	ax[3].imshow(np.abs(Plk), vmax=np.abs(Plk).max()/2, aspect='auto')
ax[0].set_title('model from above')
ax[1].set_title('model from the front')
ax[2].set_title('|Pjk|')
ax[3].set_title('|Plk|')

plt.show()
