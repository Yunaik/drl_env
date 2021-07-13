import pybullet as p
import matplotlib.pyplot as plt
import numpy as np

p.connect(p.PhysX)

p.loadURDF('urdf/laikago_description/laikago_foot.urdf', [0,0,0.47])
p.loadURDF('urdf/plane/plane.urdf')
p.setGravity(0,0, -9.81)

p.loadPlugin('eglRendererPlugin')

_,_, img, _,_ = p.getCameraImage(1920, 1080)
img = img[:,:,:3]
plt.imshow(img)
plt.show()

