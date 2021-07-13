import gym
from gym.envs.registration import registry, make, spec

def register(id,*args,**kvargs):
    if id in registry.env_specs:
        return
    else:
        return gym.envs.registration.register(id,*args,**kvargs)

#--------------- b3ph----------------------------

# register(
#     id = 'laikagoSimpleBulletEnv',
#     entry_point = 'b3px_env.singleton:laikago',
#     max_episode_steps=2000,
#     reward_threshold = 20000.0
# )

