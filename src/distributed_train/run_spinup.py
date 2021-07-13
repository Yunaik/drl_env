from rlkit.samplers.rollout_functions import rollout
from rlkit.torch.pytorch_util import set_gpu_mode
import argparse
import torch
import uuid
from rlkit.core import logger
from b3px_gym.b3px_env.examples.base_cfg import DefaultCfg
from b3px_gym.b3px_env.parallel import parallel_env_mujoco
import os
from pybullet_envs.gym_pendulum_envs import InvertedPendulumBulletEnv, InvertedPendulumSwingupBulletEnv, InvertedDoublePendulumBulletEnv
from pybullet_envs.gym_locomotion_envs import HumanoidBulletEnv, AntBulletEnv, AntBulletEnvMJC, AntBulletEnvMJC_physx, HalfCheetahBulletEnv, Walker2DBulletEnv, HopperBulletEnv
from b3px_gym.b3px_env.singleton.valkyrie_gym_env import Valkyrie
filename = str(uuid.uuid4())
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
# from rllab.envs.pybullet.valkyrie_multi_env.anymal_parallel import Anymal
from loadProgress import loadProgress
import copy
import numpy as np
#torch.cuda.set_device(0)
import sys
sys.path.insert(0, '../')
import json
from shutil import copyfile, copytree
import time

# saveVideo = True
saveVideo = False
def make_env(path):
# def make_env():
    cfg = copy.deepcopy(DefaultCfg)

    cfg['gui']          = True or saveVideo
    cfg['backend']      = 'bullet' 
    cfg['solver']       = 'tgs'
    cfg['urdf_root']    = '../urdf'
    cfg['distance']     = 3
    cfg['batch']        = 1
    cfg['gpu']          = True
    cfg['core']    = 10
    # with open('../distributed_train/robot.json') as f:

    with open('%s/robot.json'%path) as f:
       robot_config = json.load(f)

    # robot_config["robot"] = 4

    #0: HumanoidBulletEnv, 1: AntBulletEnv, 2: HalfCheetahBulletEnv, 3: Walker2DBulletEnv, 4: HopperBulletEnv

    time_to_stabilize=.0

    if robot_config["robot"] == 0:
        robot = HumanoidBulletEnv
        isMujocoEnv=True
        class_args = {"isPhysx": True if cfg['backend'] == 'physx' else False, "time_step": 0.01}
    elif robot_config["robot"] == 1:
        robot = AntBulletEnv
        isMujocoEnv=True
        class_args = {"isPhysx": True if cfg['backend'] == 'physx' else False, "time_step": 0.01}
    elif robot_config["robot"] == 2:
        robot = HalfCheetahBulletEnv
        isMujocoEnv=True
        class_args = {"isPhysx": True if cfg['backend'] == 'physx' else False, "time_step": 0.01}
    elif robot_config["robot"] == 3:
        robot = Walker2DBulletEnv
        isMujocoEnv=True
        class_args = {"isPhysx": True if cfg['backend'] == 'physx' else False, "time_step": 0.01}
    elif robot_config["robot"] == 4:
        robot = HopperBulletEnv
        isMujocoEnv=True
        class_args = {"isPhysx": True if cfg['backend'] == 'physx' else False, "time_step": 0.01}
    elif robot_config["robot"] == 5:
        robot = Valkyrie
        isMujocoEnv=False
        class_args = {"time_step": 0.005, "frame_skip": 8, "margin_in_degree": 20}
        time_to_stabilize=3.0
        print("Running Valkyrie")
    elif robot_config["robot"] == 6:
        robot = Anymal
        isMujocoEnv=False
        class_args = {"time_step": 0.001, "frame_skip": 40}
        time_to_stabilize=1.0
        print("Running ANYmal")
    env = parallel_env_mujoco.ParallelEnv(cfg, robotClass=robot, class_args=class_args, action_bandwidth=1.0, force_duration=0.,
                        isMujocoEnv=isMujocoEnv,time_to_stabilize=time_to_stabilize, spawn_height=robot_config["spawn_height"][robot_config["robot"]])
    # print("SPAWN HEIGHT: ", robot_config["spawn_height"][robot_config["robot"]])
    return env


def test_agent(env, max_path_length, ac, amount_of_tests=10):
    image_list = []
    print("env.timestep: %.4f, env.frame_skip: %.1f. Max path length: %d" % (env.timestep,env.frame_skip, max_path_length))
    o = env.reset()
    ep_rets = []
    ep_lens = []
    for _ in range(amount_of_tests):
        ep_ret = 0
        ep_len = 0

        for t in range(max_path_length):
            # print("O: ", o)
            
            # o[0][0] = 0
            # o[0][1] = 0
            # o[0][2] = 0

            # o[0][3] = 0
            # o[0][4] = 0
            # o[0][5] = -1

            # o[0][6] = 0
            # o[0][7] = 0
            # o[0][8] = 0
            a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32), deterministic=True)
            next_o, r, d, _ = env.step(a)
            # print("Obs: ", o[-6:])
            if r > -99998:
                ep_ret += r
                ep_len += 1
            o = next_o
            time.sleep(1./25)
            terminal = d 
            epoch_ended = t==max_path_length-1
            if saveVideo:
                image_list.append(env.render(distance=2, yaw=45, pitch=0, roll=90,))
            if terminal or epoch_ended:
                # print("EPOCH ENDED: ", epoch_ended, "Terminal: ", terminal)
                ep_rets.append(copy.copy(ep_ret))
                ep_lens.append(copy.copy(ep_len))
                o, ep_ret, ep_len = env.reset(), 0, 0
    frequency = 1/(env.timestep*env.frame_skip)
    return np.mean(np.array(ep_rets)), np.mean(np.array(ep_lens)), image_list, frequency

def simulate_policy():

    try:
        fpath = "/home/kai/pc_scp/pc4/log/2020-10-11_21-30-21"
        ac = torch.load('%s/pyt_save/model.pt'%fpath)
    except:
        fpath = "../data/ppo_spinup/"
        ac = torch.load('%s/pyt_save/model.pt'%fpath)
        print("PATH NOT FOUND. Running local version")
    print("Policy loaded")

    # Copying important files for reproducibility
    folder_name = time.strftime("%Y-%m-%d_%H-%M-%S")
    os.mkdir('../data/log/%s/'%folder_name)
    os.mkdir('../data/log/%s/pyt_save'%folder_name)
    copyfile(fpath+"/pyt_save/model.pt", '../data/log/%s/pyt_save/model.pt'%folder_name)
    # copyfile(fpath+"/robot.json", '../data/log/%s/robot.json'%folder_name)
    copyfile(fpath+"/progress.txt", '../data/log/%s/progress.txt'%folder_name)
    copyfile(fpath+"/vars.pkl", '../data/log/%s/vars.pkl'%folder_name)

    env = make_env(fpath)
    # env = make_env()
    
    set_gpu_mode(True)

    test_ret, test_len, image_list, step_frequency = test_agent(env, int(1/(env.timestep*env.frame_skip))*(100+3), ac, amount_of_tests=1)
    print("Test return: %.2f for %d steps" % (test_ret, test_len))
    print("=============================================")
    if saveVideo:
        clip = ImageSequenceClip(image_list, fps=step_frequency)
        clip.write_videofile('../data/log/%s/'%folder_name + '/video.mp4', fps=step_frequency, audio=False)
        print("Video saved")

    loadProgress('../data/log/%s/'%folder_name, saveFile=True)
if __name__ == "__main__":
    simulate_policy()
