import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import spinup.algos.pytorch.ppo.core as core
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
import copy

class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        # self.adv_buf_not_normalised = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        # print("SIZE: ", size)
    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        # if rew > -99998:
            # print("REW: ", rew)
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1
        # print("PTR: %d" % self.ptr)

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        # print("rews: ", rews)
        # print("vals: ", vals)
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def clean_buffer(self):
        self.ptr, self.path_start_idx = 0, 0

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        # assert self.ptr == self.max_size    # buffer has to be full before you can get
        # print("PTR AT END: %d" % self.ptr)
        self.clean_buffer()
        # the next two lines implement the advantage normalization trick
        # self.adv_buf_not_normalised = copy.copy(self.adv_buf)
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        # print("ADV MEAN: ", adv_mean)
        # print("ADV STD: ", adv_std)
        # print("Buffer size: %d" % len(self.adv_buf))
        # print("Adv: ",  (self.adv_buf))
        # assert adv_std > 1e-3, "Advmean: %.1f, adv_std: %.3f" % (adv_mean, adv_std)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}

    def combine_buffers(self, buffers):
        self.obs_buf = None
        self.act_buf = None
        self.adv_buf = None
        self.rew_buf = None
        self.ret_buf = None
        self.val_buf = None
        self.logp_buf= None
        # print("HI")
        for buffer in buffers:
            # only stack as many buffers as there are non-zero entries (use self.ptr)
            find_non_zero_idx = np.nonzero((buffer.rew_buf))[0][-1]+1
            # print("find_non_zero_idx: ", find_non_zero_idx)
            # if find_non_zero_idx == len(buffer.rew_buf):
            #     print("COOL")
            # print("Reward buffer: ", buffer.rew_buf)
            if self.obs_buf is None:
                self.obs_buf = buffer.obs_buf[:find_non_zero_idx,:] # buffer 
                self.act_buf = buffer.act_buf[:find_non_zero_idx,:] 
                self.adv_buf = buffer.adv_buf[:find_non_zero_idx] 
                self.rew_buf = buffer.rew_buf[:find_non_zero_idx] 
                self.ret_buf = buffer.ret_buf[:find_non_zero_idx] 
                self.val_buf = buffer.val_buf[:find_non_zero_idx] 
                self.logp_buf= buffer.logp_buf[:find_non_zero_idx]
            else:
                self.obs_buf = np.vstack((self.obs_buf,  buffer.obs_buf[:find_non_zero_idx, :]))
                self.act_buf = np.vstack((self.act_buf,  buffer.act_buf[:find_non_zero_idx, :]))
                self.adv_buf = np.hstack((self.adv_buf,  buffer.adv_buf[:find_non_zero_idx]))
                self.rew_buf = np.hstack((self.rew_buf,  buffer.rew_buf[:find_non_zero_idx]))
                self.ret_buf = np.hstack((self.ret_buf,  buffer.ret_buf[:find_non_zero_idx]))
                self.val_buf = np.hstack((self.val_buf,  buffer.val_buf[:find_non_zero_idx]))
                self.logp_buf= np.hstack((self.logp_buf, buffer.logp_buf[:find_non_zero_idx]))
        # assert 3 == 4 
        # assert len(self.obs_buf) >= self.max_size, "len(self.obs_buf): %d, self.max_size: %d" % (len(self.obs_buf), self.max_size)# required for get. Max size is not a strict maximum but the amount of samples we want per update

def ppo(eval_fn, sampler=None, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
        target_kl=0.01, logger_kwargs=dict(), save_freq=10):
    """
    Proximal Policy Optimization (by clipping), 

    with early stopping based on approximate KL

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with a 
            ``step`` method, an ``act`` method, a ``pi`` module, and a ``v`` 
            module. The ``step`` method should accept a batch of observations 
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``v``        (batch,)          | Numpy array of value estimates
                                           | for the provided observations.
            ``logp_a``   (batch,)          | Numpy array of log probs for the
                                           | actions in ``a``.
            ===========  ================  ======================================

            The ``act`` method behaves the same as ``step`` but only returns ``a``.

            The ``pi`` module's forward call should accept a batch of 
            observations and optionally a batch of actions, and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       N/A               | Torch Distribution object, containing
                                           | a batch of distributions describing
                                           | the policy for the provided observations.
            ``logp_a``   (batch,)          | Optional (only returned if batch of
                                           | actions is given). Tensor containing 
                                           | the log probability, according to 
                                           | the policy, of the provided actions.
                                           | If actions not given, will contain
                                           | ``None``.
            ===========  ================  ======================================

            The ``v`` module's forward call should accept a batch of observations
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``v``        (batch,)          | Tensor containing the value estimates
                                           | for the provided observations. (Critical: 
                                           | make sure to flatten this!)
            ===========  ================  ======================================


        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to PPO.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        clip_ratio (float): Hyperparameter for clipping in the policy objective.
            Roughly: how far can the new policy go from the old policy while 
            still profiting (improving the objective function)? The new policy 
            can still go farther than the clip_ratio says, but it doesn't help
            on the objective anymore. (Usually small, 0.1 to 0.3.) Typically
            denoted by :math:`\epsilon`. 

        pi_lr (float): Learning rate for policy optimizer.

        vf_lr (float): Learning rate for value function optimizer.

        train_pi_iters (int): Maximum number of gradient descent steps to take 
            on policy loss per epoch. (Early stopping may cause optimizer
            to take fewer than this.)

        train_v_iters (int): Number of gradient descent steps to take on 
            value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        target_kl (float): Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used 
            for early stopping. (Usually small, 0.01 or 0.05.)

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """
    sampler = sampler
    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    # print(locals())
    # manual_locals_key = ['actor_critic',
    #                     'ac_kwargs',
    #                     'seed',
    #                     'steps_per_epoch',
    #                     'epochs',
    #                     'gamma',
    #                     'pi_lr',
    #                     'vf_lr',
    #                     'lam',
    #                     'max_ep_len',
    #                     'logger_kwargs',
    #                     'save_freq',
    #                     'clip_ratio',
    #                     'logger',
    #                     'target_kl',
    #                     'train_pi_iters',
    #                     'train_v_iters',]
    # dict = 
    # logger.save_config(locals())

    # Random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    # expl_env = env_fn()
    eval_env = eval_fn
    obs_dim = eval_fn.observation_space.shape[0]
    act_dim = eval_fn.action_space.shape[1]
    print("OBS DIM: ", obs_dim)
    print("Act DIM: ", act_dim)
    # Create actor-critic module
    ac = actor_critic(eval_fn.observation_space, eval_fn.action_space, **ac_kwargs)

    # print("Scaling to action space")
    # SCALE According to action spae
    # Sync params across processes
    sync_params(ac)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

    # Set up experience buffer
    # print("NUM PROCS: ", num_procs())
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

    # Set up function for computing PPO policy loss
    def compute_loss_pi(data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
        # print("obs.shape: ", obs.shape)
        # print("act.shape: ", act.shape)
        # print("adv.shape: ", adv.shape)
        # print("logp_old.shape: ", logp_old.shape)
        # Policy loss
        pi, logp = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        return ((ac.v(obs) - ret)**2).mean()

    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def test_agent(env, max_path_length, amount_of_tests=1):
        o = env.reset()
        ep_rets = []
        ep_lens = []
        for _ in range(amount_of_tests):
            ep_ret = 0
            ep_len = 0

            for t in range(max_path_length):

                a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32), deterministic=True)

                next_o, r, d, _ = env.step(a)
                if r > -99998:
                    ep_ret += r
                    ep_len += 1
                o = next_o

                terminal = d 
                epoch_ended = t==max_path_length-1

                if terminal or epoch_ended:
                    ep_rets.append(copy.copy(ep_ret))
                    ep_lens.append(copy.copy(ep_len))
                    o, ep_ret, ep_len = env.reset(), 0, 0
            
        return np.mean(np.array(ep_rets)), np.mean(np.array(ep_lens))

    def update():
        # print("UPDATE")
        data = buf.get()
        # print("DATA: ", (data["obs"].shape))
        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            kl = mpi_avg(pi_info['kl'])
            if kl > 1.5 * target_kl:
                logger.log('Early stopping at step %d due to reaching max kl.'%i)
                break
            loss_pi.backward()
            mpi_avg_grads(ac.pi)    # average grads across MPI processes
            pi_optimizer.step()

        logger.store(StopIter=i)

        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            mpi_avg_grads(ac.v)    # average grads across MPI processes
            vf_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        logger.store(LossPi=pi_l_old, LossV=v_l_old,
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(loss_pi.item() - pi_l_old),
                     DeltaLossV=(loss_v.item() - v_l_old))

    # Prepare for interaction with environment
    start_time = time.time()
    # o, ep_ret, ep_len = expl_fn.reset(), 0, 0
    if sampler is not None:
        sampler.set_policy(ac)
        sampler.set_buffer(buf, buffer_class=PPOBuffer, 
                            buffer_args={   "obs_dim":obs_dim, "act_dim":act_dim, "size":local_steps_per_epoch, "gamma":gamma, "lam":lam})

    
    # print("===================================================")
    # print("MU: ", ac.get_mu())
    # print("MU params call: ", ac.getParameters(ac.get_mu()))
    # Main loop: collect experience in env and update/log each epoch
    best_ret = -1e6
    best_epoch = 0
    env_interactions = 0
    _time_for_rollout = []
    _time_for_train = []
    _time_for_test = []
    _time_for_overall = []
    for epoch in range(epochs):

        if epoch > 2000:
            save_freq = 100
        elif epoch > 1000:
            save_freq = 50
        elif epoch > 500:
            save_freq = 20
        elif epoch > 250:
            save_freq = 10

        start_overall = time.time()
        if sampler is None:
            for t in range(local_steps_per_epoch):
                a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32))
                next_o, r, d, _ = env.step(a)
                ep_ret += r
                ep_len += 1

                # save and log
                buf.store(o, a, r, v, logp)
                logger.store(VVals=v)
                
                # Update obs (critical!)
                o = next_o

                timeout = ep_len == max_ep_len
                terminal = d or timeout
                epoch_ended = t==local_steps_per_epoch-1

                if terminal or epoch_ended:
                    if epoch_ended and not(terminal):
                        print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                    # if trajectory didn't reach terminal state, bootstrap value target
                    if timeout or epoch_ended:
                        _, v, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
                    else:
                        v = 0
                    buf.finish_path(v)
                    if terminal:
                        # only save EpRet / EpLen if trajectory finished
                        logger.store(EpRet=ep_ret, EpLen=ep_len)
                    o, ep_ret, ep_len = env.reset(), 0, 0
        else:
            start_collect = time.time()
            paths, buf = sampler.obtain_ppo_samples()
            _time_for_rollout.append(time.time()-start_collect)
            # print("PATHS: ", paths)
            for path in paths:
                ep_len = 0
                ep_ret = 0
                for reward in path["rewards"]:
                    ep_len += 1
                    ep_ret += reward
                    env_interactions+=1
                # print("Return: %.1f, len: %d" % (ep_ret, ep_len))
                logger.store(EpRet=ep_ret, EpLen=ep_len)
                
            logger.store(EnvInteractions=env_interactions)
        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            logger.save_state({}, itr=epoch)

        # Perform PPO update!
        start_train = time.time()
        update()
        _time_for_train.append(time.time()-start_train)
        # TODO write test here!

        start_test = time.time()
        if ((epoch % save_freq) == 0): # and (epoch > 1): # or epoch < 2:# 
            # print("EPOCH: %d" % epoch)
            test_ep_ret, test_ep_len= test_agent(eval_env, max_ep_len)
            if test_ep_ret > best_ret:
                best_ret = test_ep_ret
                logger.save_state({}, itr=None)
                best_epoch = epoch

            logger.store(TestEpRet=test_ep_ret)
            logger.store(TestEpLen=test_ep_len)
            logger.store(BestEpoch=best_epoch)
            
            _time_for_test.append(time.time()-start_test)
            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('BestEpoch',    average_only=True)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpRet', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('EnvInteractions', average_only=True)
            # logger.log_tabular('VVals', with_min_and_max=True)
            # logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
            # logger.log_tabular('LossPi', average_only=True)
            # logger.log_tabular('LossV', average_only=True)
            # logger.log_tabular('DeltaLossPi', average_only=True)
            # logger.log_tabular('DeltaLossV', average_only=True)
            # logger.log_tabular('Entropy', average_only=True)
            # logger.log_tabular('KL', average_only=True)
            # logger.log_tabular('ClipFrac', average_only=True)
            # logger.log_tabular('StopIter', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            _time_for_overall.append(time.time()-start_overall)
            # Only dump when save freq met
            logger.dump_tabular()
            print("Rollout time: %.4fs, Train time: %.4fs, Test time: %.4fs, Overall time: %.4fs" % (np.mean(np.array(_time_for_rollout)),
                                                                                                np.mean(np.array(_time_for_train)),
                                                                                                np.mean(np.array(_time_for_test)),
                                                                                                np.mean(np.array(_time_for_overall))))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='ppo')
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    ppo(lambda : gym.make(args.env), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma, 
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
        logger_kwargs=logger_kwargs)