from collections import deque, OrderedDict

import numpy as np
import time
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.data_management.path_builder import PathBuilder
from rlkit.samplers.data_collector.base import StepCollector

from multiprocessing.dummy import Pool as ThreadPool
import multiprocessing
from itertools import product
import copy
import torch
class BatchMdpStepCollector_parallel(StepCollector):
    def __init__(
            self,
            batch_collector_list, #gets a list of BatchMdpStepCollector
            max_num_epoch_paths_saved=None,
            n_workers=1,
            max_path_length=None,
            num_steps=None
    ):
        self.max_path_length=max_path_length
        self.num_steps=num_steps
        self.n_workers = n_workers
        self.pool = ThreadPool(n_workers)
        # self.pool = multiprocessing.Pool(processes=n_workers)
        self.batch_collector_list = batch_collector_list
        self._max_num_epoch_paths_saved = max_num_epoch_paths_saved
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self.buffer=None

    def get_epoch_paths(self):
        return self._epoch_paths

    def set_policy(self, policy):
        for batch_collector in self.batch_collector_list:
            batch_collector.set_policy(policy)

    def set_buffer(self, buffer, buffer_class=None, buffer_args=None):
        self.buffer = buffer

        for batch_collector in self.batch_collector_list:
            batch_collector.set_buffer(buffer_class, buffer_args)        

    def obtain_samples(self):
        assert self.max_path_length is not None
        self.collect_new_steps(self.max_path_length, self.num_steps, isPPO=False)
        paths = copy.copy(self.get_epoch_paths())
        self.end_epoch(epoch=0)
        return paths

    def obtain_ppo_samples(self):
        assert self.max_path_length is not None
        self.collect_new_steps(self.max_path_length, self.num_steps, isPPO=True)
        paths = copy.copy(self.get_epoch_paths())
        self.end_epoch(epoch=0)
        return paths, self.buffer

    def end_epoch(self, epoch):
        # print("Path len: ", len(self._epoch_paths))
        # print("================ENDING EPOCH")
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        for batch_collector in self.batch_collector_list:
            batch_collector.end_epoch(epoch)

    def collect_parallel_paths(self, batch_mdp_step_collector):

        batch_mdp_step_collector.collect_new_steps(
                                max_path_length=self.collect_new_steps_dict["max_path_length"],
                                num_steps=self.collect_new_steps_dict["num_steps"],
                                discard_incomplete_paths=self.collect_new_steps_dict["discard_incomplete_paths"],
                                isPPO=self.collect_new_steps_dict["isPPO"],
                                )
        # print("GET BUFFER: ", batch_mdp_step_collector.get_buffer())
        if self.collect_new_steps_dict["isPPO"]:
            return batch_mdp_step_collector.get_epoch_paths(), batch_mdp_step_collector.get_buffer()
        else:
            return batch_mdp_step_collector.get_epoch_paths()
    def collect_new_steps(
            self,
            max_path_length,
            num_steps,
            discard_incomplete_paths=False,
            isPPO=True
    ):
        # print("Collecting %d steps with max %d " % (num_steps, max_path_length))
        self.collect_new_steps_dict = {"max_path_length":max_path_length, "num_steps": num_steps//self.n_workers, "discard_incomplete_paths": discard_incomplete_paths, "isPPO": isPPO}
        # print("Collect new steps dict: ", self.collect_new_steps_dict)
        results = self.pool.map(self.collect_parallel_paths, self.batch_collector_list)
        if isPPO:
            buffers = []
            paths   = []
            for result in results:
                paths.append(result[0])
                buffers.append(result[1])
            for path in paths:
                self._epoch_paths += path
            if isPPO:
                self.buffer.combine_buffers(buffers)
        else:
            for path in results:
                self._epoch_paths += path
                
    def close_pool(self):
        self.pool.close()
        self.pool.join()

class BatchMdpStepCollector(StepCollector):
    def __init__(
            self,
            env,
            policy,
            max_path_length,
            max_num_epoch_paths_saved=None,
            render=False,
            render_kwargs=None,
    ):
        if render_kwargs is None:
            render_kwargs = {}
        self._env = env
        self._policy = policy
        self.buffers = None
        self.buffer_combined = None
        self._max_num_epoch_paths_saved = max_num_epoch_paths_saved
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._render = render
        self._render_kwargs = render_kwargs

        self._num_steps_total = 0
        self._num_paths_total = 0
        self._obs = None  # cache variable

        self._max_path_length = max_path_length
        self.ct = {}
        self.ct['obs'] = np.zeros((max_path_length, self._env.batch, self._env.observation_space.shape[0]))
        self.ct['act'] = np.zeros((max_path_length, self._env.batch, self._env.action_space.shape[1]))
        self.ct['rewards'] = np.zeros((max_path_length, self._env.batch, 1))
        self.ct['terminal'] = np.zeros((max_path_length, self._env.batch, 1), dtype= np.uint8)
        self.ct['agent_infos'] = None
        self.ct['env_infos'] = None
        # ct['idy_const'] = np.arange(self._env.batch)
        self.idy_const = np.arange(self._env.batch)

    def get_epoch_paths(self):
        return self._epoch_paths

    def get_buffer(self):
        # print("IN ONE BUFFER: ", self.buffers[0].act_buf.shape)
        # print("IN TWO BUFFER: ", self.buffers[1].act_buf.shape)
        self.buffer_combined.combine_buffers(self.buffers)
        for buffer in self.buffers:
            buffer.clean_buffer()
        # print("Buffer: ", buffer)
        return self.buffer_combined


    def end_epoch(self, epoch):
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._obs = None

    def set_policy(self, policy):
        self._policy = policy

    def set_buffer(self, buffer_class, buffer_args):
        self.buffers = []
        for idx in range(self._env.batch):
            self.buffers.append(buffer_class(**buffer_args))
        self.buffer_combined = buffer_class(**buffer_args)
        
    def get_diagnostics(self):
        path_lens = [len(path['actions']) for path in self._epoch_paths]
        stats = OrderedDict([
            ('num steps total', self._num_steps_total),
            ('num paths total', self._num_paths_total),
        ])
        stats.update(create_stats_ordered_dict(
            "path length",
            path_lens,
            always_show_all_stats=True,
        ))
        return stats

    def get_snapshot(self):
        return dict(
            # env=self._env,
            policy=self._policy,
        )

    def collect_new_steps(
            self,
            max_path_length,
            num_steps,
            discard_incomplete_paths=False,
            isPPO=False
    ):  

        # print("Subsampler collecting %d steps" % num_steps)
        # while num_steps >= 0:
        samples = 0
        for _ in range(int(np.ceil(num_steps/self._env.batch))):
        # for _ in range(num_steps):
            self.collect_one_step(max_path_length, isPPO=isPPO)
            # num_steps -= self._env.batch
            samples += self._env.batch
            # print("Samples: %d" % samples)
        # print("Num steps: %d, batch: %d, iters: %.3f" % (num_steps, self._env.batch, int(np.ceil(num_steps/self._env.batch))))
        # print("BATCH SIZE: %d / %d " % (, num_steps))
        self._force_flash()
        # print("PTR: %d" % self.buffers[0].ptr)

    def collect_one_step(
            self,
            max_path_length,
            discard_incomplete_paths=False,
            isPPO=False
    ):
        if isPPO:

            if self._obs is None:
                self._start_new_rollout([]) # setting self._obs in here through reset
            # start_time = time.time()
            action, v, logp = self._policy.step(torch.as_tensor(self._obs, dtype=torch.float32))

            # print("Obs: ", self._obs.shape)
            # actions = []
            # for idx in range(self._env.batch):
                # print("OBS: ", self._obs[idx, :])
                # action, v, logp = self._policy.step(torch.as_tensor(self._obs[idx, :], dtype=torch.float32))
                # print("ACTION: ", action)
            #     actions.append(action)
            # print("ACTION: ", np.array(actions).shape)
            # print("END TIME: %.4fms" % ((time.time()-start_time)*1000))
            next_ob, reward, terminal, env_info = (
                self._env.step(action)
            )                
            
            for idx, buffer in enumerate(self.buffers):
                # print("IDX: %d" % idx)
                if reward[idx] > -99998:
                    # print("reward[idx]: ", reward[idx])
                    buffer.store(self._obs[idx], action[idx], reward[idx], v[idx], logp[idx])

            if self._render:
                self._env.render(**self._render_kwargs)

            if reward[0] > -99998:
                self.ct['obs'][self._env._env_step_counter-1, self.idy_const] = self._obs
                self.ct['act'][self._env._env_step_counter-1, self.idy_const] = action
                # print("Reward shape: ", reward.shape)
                self.ct['rewards'][self._env._env_step_counter-1, self.idy_const] = reward.reshape(-1, 1)
                self.ct['terminal'][self._env._env_step_counter-1, self.idy_const] = terminal.reshape(-1, 1)
                # print("Reward: ", reward)
            # print("Env counter: %d, maxpathlength: %d" % (self._env._env_step_counter[0], max_path_length))

            if np.any(terminal) or np.any(self._env._env_step_counter >= max_path_length):

                id1 = np.where(terminal)[0]
                id2 = np.where(self._env._env_step_counter >= max_path_length)[0]
                ids = np.union1d(id1, id2)
                # print("Termin: ", np.any(terminal), "Over: ", np.any(self._env._env_step_counter >= max_path_length))

                for idx in id1:
                    v = 0 # value is 0 if terminated
                    # print("FINISHING PATH because terminated")
                    self.buffers[idx].finish_path(v) # can't run this outside, because this idx is different than id2

                for idx in id2:
                    _, v, _ = self._policy.step(torch.as_tensor(next_ob, dtype=torch.float32)) # get the value for next step
                    # print("FINISHING PATH because end reached")
                    self.buffers[idx].finish_path(v[idx])

                self._handle_rollout_ending(ids, next_ob)
                self._start_new_rollout(ids)
            else:
                self._obs = next_ob

        
        else:
            if self._obs is None:
                self._start_new_rollout([])
            # print("Obs: ", self._obs.shape)
            action, agent_info = self._policy.get_action(self._obs)
            # print("ACTION: ", action.shape)
            next_ob, reward, terminal, env_info = (
                self._env.step(action)
            )
            
            if self._render:
                self._env.render(**self._render_kwargs)
            if reward[0] > -99998:
                self.ct['obs'][self._env._env_step_counter-1, self.idy_const] = self._obs
                self.ct['act'][self._env._env_step_counter-1, self.idy_const] = action
                # print("Reward shape: ", reward.shape)
                self.ct['rewards'][self._env._env_step_counter-1, self.idy_const] = reward.reshape(-1, 1)
                self.ct['terminal'][self._env._env_step_counter-1, self.idy_const] = terminal.reshape(-1, 1)
            # print("next_ob: ", next_ob.shape)
            if np.any(terminal) or np.any(self._env._env_step_counter >= max_path_length):

                id1 = np.where(terminal)[0]
                id2 = np.where(self._env._env_step_counter >= max_path_length)[0]
                ids = np.union1d(id1, id2)
                # print("Termin: ", np.any(terminal), "Over: ", np.any(self._env._env_step_counter >= max_path_length))
                # print("IDs:" , ids)
                self._handle_rollout_ending(ids, next_ob)
                self._start_new_rollout(ids)
            else:
                self._obs = next_ob

    def _start_new_rollout(self, ids=[]):
        self._obs = self._env.reset(ids)

    def _handle_rollout_ending(
            self,
            ids,
            next_ob
    ):
        top = self._env._env_step_counter
        for id in ids:
            # print("ID: %d" % id)
            path = dict(
                observations=self.ct['obs'][:top[id], id].copy(),
                actions=self.ct['act'][:top[id], id].copy(),
                rewards=self.ct['rewards'][:top[id], id].copy(),
                terminals=self.ct['terminal'][:top[id], id].copy(),
            )
            path['next_observations'] = np.vstack(
                (
                    path['observations'][1:].copy(),
                    next_ob[id].reshape(1, -1)
                )
            ) if path['observations'].shape[0] != 1 else next_ob[id].reshape(1, -1).copy()
            path['agent_infos'] = [{}] * path['rewards'].size
            path['env_infos'] = [{}] * path['rewards'].size

            self._epoch_paths.append(path)
            self._num_paths_total += 1
            self._num_steps_total += path['rewards'].size
        # print("NUM STEP TOTAL: %d" % self._num_steps_total)
    def _force_flash(self):
        # Force Clear Cache Data in one epoch
        ids = np.where(self._env._env_step_counter != 0)[0]
        self._handle_rollout_ending(ids, self._obs)



class Random_BatchMdpStepCollector(StepCollector):
    def __init__(
            self,
            env,
            action,
            max_path_length,
            max_num_epoch_paths_saved=None,
            render=False,
            render_kwargs=None,
    ):
        if render_kwargs is None:
            render_kwargs = {}
        self._env = env
        self._action = action
        self._max_num_epoch_paths_saved = max_num_epoch_paths_saved
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._render = render
        self._render_kwargs = render_kwargs

        self._num_steps_total = 0
        self._num_paths_total = 0
        self._obs = None  # cache variable

        self._max_path_length = max_path_length
        self.ct = {}
        self.ct['obs'] = np.zeros((max_path_length, self._env.batch, self._env.observation_space.shape[0]))
        self.ct['act'] = np.zeros((max_path_length, self._env.batch, self._env.action_space.shape[1]))
        self.ct['rewards'] = np.zeros((max_path_length, self._env.batch, 1))
        self.ct['terminal'] = np.zeros((max_path_length, self._env.batch, 1), dtype= np.uint8)
        self.ct['agent_infos'] = None
        self.ct['env_infos'] = None
        # ct['idy_const'] = np.arange(self._env.batch)
        self.idy_const = np.arange(self._env.batch)

    def get_epoch_paths(self):
        return self._epoch_paths

    def end_epoch(self, epoch):
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._obs = None

    def get_diagnostics(self):
        path_lens = [len(path['actions']) for path in self._epoch_paths]
        stats = OrderedDict([
            ('num steps total', self._num_steps_total),
            ('num paths total', self._num_paths_total),
        ])
        stats.update(create_stats_ordered_dict(
            "path length",
            path_lens,
            always_show_all_stats=True,
        ))
        return stats

    def get_snapshot(self):
        return dict(
            # env=self._env,
            policy=self._policy,
        )

    def collect_new_steps(
            self,
            max_path_length,
            num_steps,
            discard_incomplete_paths=False
    ):
        while num_steps >= 0:
        # for _ in range(num_steps):
            self.collect_one_step(max_path_length)
            num_steps -= self._env.batch

        self._force_flash()


    def collect_one_step(
            self,
            max_path_length,
    ):
        if self._obs is None:
            self._start_new_rollout([])

        # action, agent_info = self._policy.get_action(self._obs)
        # print("Action: ", self._action.shape)
        next_ob, reward, terminal, env_info = (
            self._env.step(self._action)
        )

        if self._render:
            self._env.render(**self._render_kwargs)

        self.ct['obs'][self._env._env_step_counter-1, self.idy_const] = self._obs
        self.ct['act'][self._env._env_step_counter-1, self.idy_const] = self._action
        self.ct['rewards'][self._env._env_step_counter-1, self.idy_const] = reward.reshape(-1, 1)
        self.ct['terminal'][self._env._env_step_counter-1, self.idy_const] = terminal.reshape(-1, 1)

        if np.any(terminal) or np.any(self._env._env_step_counter >= max_path_length):

            id1 = np.where(terminal)[0]
            id2 = np.where(self._env._env_step_counter >= max_path_length)[0]
            ids = np.union1d(id1, id2)

            self._handle_rollout_ending(ids, next_ob)
            self._start_new_rollout(ids)
        else:
            self._obs = next_ob

    def _start_new_rollout(self, ids=[]):
        self._obs = self._env.reset(ids)

    def _handle_rollout_ending(
            self,
            ids,
            next_ob
    ):
        top = self._env._env_step_counter
        for id in ids:
            path = dict(
                observations=self.ct['obs'][:top[id], id].copy(),
                actions=self.ct['act'][:top[id], id].copy(),
                rewards=self.ct['rewards'][:top[id], id].copy(),
                terminals=self.ct['terminal'][:top[id], id].copy(),
            )
            path['next_observations'] = np.vstack(
                (
                    path['observations'][1:].copy(),
                    next_ob[id].reshape(1, -1)
                )
            ) if path['observations'].shape[0] != 1 else next_ob[id].reshape(1, -1).copy()
            path['agent_infos'] = [{}] * path['rewards'].size
            path['env_infos'] = [{}] * path['rewards'].size

            self._epoch_paths.append(path)
            self._num_paths_total += 1
            self._num_steps_total += path['rewards'].size

    def _force_flash(self):
        # Force Clear Cache Data in one epoch
        ids = np.where(self._env._env_step_counter != 0)[0]
        self._handle_rollout_ending(ids, self._obs)
