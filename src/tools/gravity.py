import pybullet as p
import pybullet_data as pd
import time
import math

usePhysX = False
useMaximalCoordinates = True

if usePhysX:
  p.connect(p.PhysX,options="--numCores=8 --gpu=1 --solver=tgs")
  p.loadPlugin("eglRendererPlugin")
else:
  p.connect(p.DIRECT)

p.setPhysicsEngineParameter(fixedTimeStep=1./1000.,numSolverIterations=10, minimumSolverIslandSize=1024)
p.setPhysicsEngineParameter(contactBreakingThreshold=0.01)

p.setAdditionalSearchPath(pd.getDataPath())
p.setGravity(0,0, -9.81)
#Always make ground plane maximal coordinates, to avoid performance drop in PhysX
#See https://github.com/NVIDIAGameWorks/PhysX/issues/71

p.loadURDF("plane.urdf", useMaximalCoordinates=True)#useMaximalCoordinates)

id = p.loadURDF("../urdf/laikago_description/laikago_foot.urdf", [0,0, 0.55])

print(p.getDynamicsInfo(id, 1))

i = 0
while True:
  p.stepSimulation()
  vel, agl = p.getBaseVelocity(id)
  pos, _ = p.getBasePositionAndOrientation(id)

  i+=1
  print(i, ':', pos, vel)