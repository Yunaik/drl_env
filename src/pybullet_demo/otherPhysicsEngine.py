import pybullet as p
import pybullet_data as pd
import time
import math

usePhysX = True
useMaximalCoordinates = True
if usePhysX:
  p.connect(p.PhysX,options="--numCores=8 --solver=pgs")
  # p.loadPlugin("eglRendererPlugin")
else:
  p.connect(p.GUI)

p.setPhysicsEngineParameter(fixedTimeStep=1./1000.,numSolverIterations=1000, minimumSolverIslandSize=1024)
p.setPhysicsEngineParameter(contactBreakingThreshold=0.001)

p.setAdditionalSearchPath(pd.getDataPath())

p.loadURDF("plane.urdf", useMaximalCoordinates=True)#useMaximalCoordinates)
p.loadURDF("franka_model/arm.urdf")
# p.loadURDF("/home/syslot/DevSpace/bullet3/data/humanoid.urdf")


print("loaded!")




p.setGravity(0,0,-10)

count=0
while (1):
  count+=1
  if (count==12):
      p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)

  
  p.stepSimulation()

  if count > 10*1000:
    break

p.disconnect()
