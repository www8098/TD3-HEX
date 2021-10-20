import gym
import numpy as np

# https://github.com/openai/gym/blob/master/gym/core.py


class NormalizedEnv(gym.ActionWrapper):
    """ Wrap ActionNoise """
    def __init__(self, env):
        super().__init__(env)

    def action(self, action):
        act_k = (self.action_space.high - self.action_space.low)/ 2.
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k * action + act_b

    def reverse_action(self, action):
        act_k_inv = 2./(self.action_space.high - self.action_space.low)
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k_inv * (action - act_b)
    

class reward_clip(gym.RewardWrapper):
    """ Wrap reward """
    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward):
        if reward == -100:
            return -1
        else:
            return reward


class observation_norm(gym.ObservationWrapper):
    """ observation normalized """
    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):
        obs_h = self.observation_space.high
        obs_l = self.observation_space.low
        return (obs - obs_l) / (obs_h - obs_l)
