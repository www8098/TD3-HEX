
import numpy as np
import copy
import matplotlib.pyplot as plt
import scipy.io
from memory import RingBuffer
from util import duplicate_action, print_yellow

# [reference]: https://github.com/ghliu/pytorch-ddpg/master/main


class Evaluator(object):

    def __init__(self, episodes_num, interval, mode, save_path='', episodes_length=None):
        self.episodes_num = episodes_num
        self.interval = interval
        self.mode = mode
        self.save_path = save_path
        self.episodes_length = episodes_length
        
        self.res = np.array([]).reshape(episodes_num, 0)

    def __call__(self, env, policy, window_length, debug=False, visualize=False, save=True):

        self.is_training = False
        result = list()
        ob_buf = RingBuffer(window_length)
        observation = env.reset()
        for i in range(window_length):
            ob_buf.append(copy.deepcopy(observation))

        for episode in range(self.episodes_num):
            # reset at the start of episode
            observation = env.reset()
            ob_buf.append(copy.deepcopy(observation))
            episode_steps = 0
            episode_reward = 0.
                
            assert observation is not None

            # start episode
            done = False
            while not done:
                # basic operation, ActionNoise ,reward, blablabla ...
                if window_length == 1:
                    action = policy(observation)
                else:
                    ob = []
                    for i in range(window_length):
                        ob.extend(ob_buf[i])
                    action = policy(ob)

                if self.mode == 'COUPLE':
                    observation, reward, done, info = env.step(duplicate_action(action))
                else:
                    observation, reward, done, info = env.step(action)

                ob_buf.append(copy.deepcopy(observation))
                
                if self.episodes_length and episode_steps >= self.episodes_length -1:
                    done = True
                
                if visualize:
                    env.render(mode='human')

                # update
                episode_reward += reward
                episode_steps += 1

            if debug: print_yellow('[Evaluate] #Episode{}: episode_reward:{}'.format(episode,episode_reward))
            result.append(episode_reward)

        result = np.array(result).reshape(-1,1)
        self.res = np.hstack([self.res, result])

        if save:
            self.save_results('{}/validate_reward'.format(self.save_path))
        return np.mean(result)

    def save_results(self, fn):

        y = np.mean(self.res, axis=0)
        error=np.std(self.res, axis=0)
                    
        x = range(0,self.res.shape[1]*self.interval, self.interval)
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        plt.xlabel('Time step')
        plt.ylabel('Average Reward')
        ax.errorbar(x, y, yerr=error, fmt='-o')
        plt.savefig(fn+'.png')
        scipy.io.savemat(fn+'.mat', {'reward': self.res})
