import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete
import copy
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):
    
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, action_space=None):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)
        # if action_space is not None:
        #     for idx in range(act_dim):
        #         self.mu_net[idx] = (self.mu_net[idx]+1.0)*(action_space.high[idx]-action_space.low[idx])/2+action_space.low[idx] # scale by action boundaries


    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution


class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.



class MLPActorCritic(nn.Module):


    def __init__(self, observation_space, action_space, 
                 hidden_sizes=(64,64), activation=nn.Tanh, squash_output=False):
        super().__init__()

        obs_dim = observation_space.shape[0]
        # print("OBS DIM: ", observation_space.shape)
        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[1], hidden_sizes, activation, action_space)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)
        self.action_space = action_space
        # build value function
        self.v  = MLPCritic(obs_dim, hidden_sizes, activation)
        self.action_tan = None
        self.squash_output = squash_output
        if self.squash_output:
            print("Scaling all actions to action_space low: ", action_space.low)
            print("Scaling all actions to action_space high: ", action_space.high)

    def get_mu(self):
        return self.pi.mu_net

    def getParameters(self, model):
        names = []
        params = []
        for name, param in model.named_parameters():
            names.append(name)
            params.append(param.detach().numpy())

        # print("Names: ", names)
        # print("Params: ", params)
        # assert 3 == 4
        return names, params
    def step(self, obs, deterministic=False):
        with torch.no_grad():

            pi = self.pi._distribution(obs)
            if deterministic:
                a = self.pi.mu_net(obs)
            else:
                a = pi.sample()


            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
            # print("Unsquashed: ", a.numpy())
            # self.action_tan = copy.copy(action)
            # action = np.array(copy.copy(a.numpy()))
            # print("Action raw:", action)

            if self.squash_output:

                action = np.tanh(copy.copy(a.numpy()))
                self.action_tan = copy.copy(action)
                # print("Action tanh", self.action_tan)
                action = (action+1.0)*(np.array(self.action_space.high)-np.array(self.action_space.low))/2+np.array(self.action_space.low) # scale by action boundaries
                # print("Action scaled:", action)
            else:
                action = a.numpy()
        return action, v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]