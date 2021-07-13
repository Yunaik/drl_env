This is the distributed simulation environment code accompanying the paper "Fast Sample Collection Through Massive Parallelisation For Accelerated Deep Reinforcement Learning Development"

# Installation

Please clone the bullet3 fork https://github.com/syslot/bullet3 on branch physx4.0 and build it according to the instructions

# Example training process

Example training processes can be found in `src/Train` for the Laikago robot

The training for ANYmal, Valkyrie, and the Mujoco environments can be found in `src/distributed_train`. The script `train_all.py` specifies the training parameters and training procedures as described in the paper.
