import numpy as np
import sys
from silx.gui import qt
from silx.gui.plot3d.SceneWindow import SceneWindow
from silx.gui.colors import Colormap

# load the data and calculate the strain
data = np.load('rectified.npz')['data']
psize = np.load('rectified.npz')['psize']

# crop
N = data.shape[0]
n = 10
data = data[N//2-n:N//2+n, N//2-n:N//2+n, N//2-n:N//2+n]

# optionally convert phase to duz/dz
phase = np.angle(data)
strain = np.zeros_like(phase)
for j in range(1, strain.shape[1]-1):
    strain[:, j, :] = (phase[:, j+1, :] - phase[:, j-1, :]) / 2
data_dz = np.abs(data) * np.exp(1j * strain)

# experiment: convert phase to duz/dR (radial derivative)
com = np.sum(np.indices(data.shape) * np.abs(data), axis=(1,2,3)) / np.sum(np.abs(data))
dxyz = np.indices(data.shape) - com.reshape((3,1,1,1))
R = np.sqrt(np.sum(dxyz**2, axis=0)) * psize * 1e9 # nm
print(R.min(), R.max())
strain = np.zeros_like(data, dtype=float)
for i in range(1, data.shape[0]-1):
    for j in range(1, data.shape[1]-1):
        for k in range(1, data.shape[2]-1):
            du = np.angle(data[i, j, k]) - np.angle(data[i-1:i+2, j-1:j+2, k-1:k+2])
            dR = R[i, j, k] - R[i-1:i+2, j-1:j+2, k-1:k+2]
            #use = np.where(~np.isclose(dR, 0))
            use = np.where(dR > .5)
            du = du[use].flatten()
            dR = dR[use].flatten()
            strain[i, j, k] = np.sum(du / dR) / len(du)
data_dR = np.abs(data) * np.exp(1j * strain)

# Create a SceneWindow widget in an app
app = qt.QApplication([])
window = SceneWindow()

# Get the SceneWidget contained in the window and set its colors
widget = window.getSceneWidget()
widget.setBackgroundColor((1., 1., 1., 1.))
widget.setForegroundColor((1., 1., 1., 1.)) # the box color
widget.setTextColor((0, 0, 0, 1.))

# change the camera angle, there are no silx API calls for this...
# direction is the line of sight of the camera and up is the direction
# pointing upward in the screen plane 
# from experimentation these axes are are [y z x]
widget.viewport.camera.extrinsic.setOrientation(direction=[0.9421028 , 0.0316029 , 0.33383173],
                                                up=[-0.03702362,  0.99926555,  0.00988633])
widget.centerScene()

# add the volume, which will be made complex based on data.dtype
volume = widget.addVolume(data_dz)
volume.setScale(*(1e9*psize,)*3)

# our array is indexed as x z y, with z indexed backwards
# they expect the data as z y x.
# it might be flipped somehow but the axes are correct now
widget._sceneGroup.setAxesLabels(xlabel='Y', ylabel='Z', zlabel='X')

# add and manipulate an isosurface, of type ComplexIsosurface
volume.addIsosurface(np.abs(data).max()/4, 'r')
iso = volume.getIsosurfaces()[0]
iso.setComplexMode('phase')
iso.setColormap(Colormap('jet', vmin=-3.14/10, vmax=3.14/10))

# disable the default cut plane
cuts = volume.getCutPlanes()
[cut.setVisible(False) for cut in cuts]

window.show()

# Display exception in a pop-up message box
sys.excepthook = qt.exceptionHandler

# Run Qt event loop
app.exec_()
