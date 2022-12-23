"""
Makes the little 3d rendering in Fig 1.
"""

import numpy as np
import sys
from silx.gui import qt
from silx.gui.plot3d.SceneWindow import SceneWindow, items
from silx.gui.colors import Colormap

# load the data and calculate the strain
data = np.load('rectified.npz')['data']
psize = np.load('rectified.npz')['psize'][0]

# crop?
N = data.shape[0]
#n = 10
#data = data[N//2-n:N//2+n, N//2-n:N//2+n, N//2-n:N//2+n]

# optionally convert phase to duz/dz
phase = np.angle(data)
strain = np.zeros_like(phase)
for j in range(1, strain.shape[1]-1):
    strain[:, j, :] = (phase[:, j+1, :] - phase[:, j-1, :]) / 2
data_dz = np.abs(data) * np.exp(1j * strain)
print(np.angle(data))

# Create a SceneWindow widget in an app
app = qt.QApplication([])
window = SceneWindow()

# Get the SceneWidget contained in the window and set its colors
widget = window.getSceneWidget()
widget.setBackgroundColor((1., 1., 1., 1.))
widget.setForegroundColor((1., 1., 1., 1.)) # the box color
widget.setTextColor((0, 0, 0, 1.))

# change the camera angle, there are/were no silx API calls for this...
widget.viewport.camera.extrinsic.setOrientation(
    direction=[0. , 0. , 1.],
    up=[0.,  1.,  .1]
)
widget.centerScene()

# Create a group in which to put stuff with shared transforms
group = items.GroupItem()

# make a volume
volume = items.ComplexField3D()
volume.setData(data)
# volume.setScale(*(1e9*psize,)*3)
volume.addIsosurface(np.abs(data).max()/5, 'r')
# remove default cut planes
planes = volume.getCutPlanes()
[p.setVisible(False) for p in planes]

# try a clipping plane
clipPlane = items.ClipPlane()  # Create a new clipping plane item
clipPlane.setNormal((0., 0., 1.))  # Set its normal
clipPlane.setPoint((0., 0., N / 2))  # Set a point on the plane


# group.addItem(clipPlane)
group.addItem(volume)
widget.addItem(group)
window.show()

# Display exception in a pop-up message box
sys.excepthook = qt.exceptionHandler

# Run Qt event loop
app.exec_()
