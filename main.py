import numpy as np
import argparse
from copy import deepcopy
import torch
import gym
from gym.wrappers import TimeLimit
from tensorboardX import SummaryWriter

from normalized_env import *
from evaluator import Evaluator
from td3 import TD3

from parameters import get_args
from trainer import *

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

MODE = 'COUPLE'

if __name__ == "__main__":
    args = get_args()
    if args.mode != 'BC+FineTune':
        # args.output = get_output_folder(args.output, args.env)
        args.output = get_output_folder(args.output, 'kraby')
    else:
        args.output = get_output_folder('bc_output', args.env)
        
    if args.resume == 'default':
        if MODE == 'COUPLE':
            args.resume = 'output/{}-run4'.format('kraby')
        else:
            args.resume = 'output/{}-normal'.format('kraby')

#  C:\Users\ASUS\AppData\Roaming\Python\Python37\site-packages\gym_kraby\envs
    if args.mode == 'train':
        env = gym.make(args.env, render=False)
    else:
        env = gym.make(args.env, render=True)
    # env = TimeLimit(env, 32)
    # env = NormalizedEnv(env)
    # env = reward_clip(env)
    # go to gym/env/box2d/nipedal_walker to change the probability
    
    writer = SummaryWriter(log_dir='{}/tensorboardx_log'.format(args.output))

    if args.seed > 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        env.seed(args.seed)
        env.action_space.np_random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    nb_states = env.observation_space.shape[0] * args.window_length
    if MODE == 'COUPLE':
        nb_actions = 7
    else:
        nb_actions = env.action_space.shape[0]

    print(nb_actions)

    agent = TD3(nb_states, nb_actions, args, noise=False)
    evaluate = Evaluator(args.validate_episodes, 
        args.validate_steps, MODE, args.output, episodes_length=args.max_episode_length)

    # agent.load_weights(args.resume)
    # agent.load_weights('bc_output/Walker2d-v2')

    if args.mode == 'train':
        train(writer, args, agent, env, evaluate, MODE,
            debug=args.debug, num_interm=args.num_interm, visualize=False)

    elif args.mode == 'test':
        test(writer, args.validate_episodes, agent, env, args.window_length, evaluate, args.resume,
            visualize=True, debug=args.debug)
    
    elif args.mode == 'BC+FineTune':
        bc = behavior_clone(args, nb_states, nb_actions)
        bc.clone(agent, 100000)

    else:
        raise RuntimeError('undefined mode {}'.format(args.mode))
