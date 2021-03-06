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

MODE = 'ROTATION'
DIRECTION = 'BACKWARD' # in ['FORWARD', 'BACKWARD', 'LEFT', 'RIGHT']
if __name__ == "__main__":
#%% import the arguments
    args = get_args()
    args.method = MODE

    if args.mode != 'BC+FineTune':
        # args.output = get_output_folder(args.output, args.env)
        args.output = get_output_folder(args.output, 'kraby-{}'.format(MODE))
    else:
        args.output = get_output_folder('bc_output', args.env)

    if args.resume == 'default':
        if args.method == 'COUPLE':
            args.resume = 'output/{}-couple'.format('kraby')
        elif args.method == 'NORMAL':
            args.resume = 'output/{}-normal'.format('kraby')
        elif args.method == 'ROTATION':
            args.resume = 'output/{}-rotation'.format('kraby')
        elif args.method == 'AUTO':
            args.resume == 'output/{}-auto'.format('kraby')

#%% normalized environemnt
    # C:\Users\ASUS\AppData\Roaming\Python\Python37\site-packages\gym_kraby\envs
    if args.mode == 'train':
        env = gym.make(args.env, render=False, direction=DIRECTION)
    else:
        env = gym.make(args.env, render=True, direction=DIRECTION)
    # env = TimeLimit(env, 32)
    # env = NormalizedEnv(env)
    # env = reward_clip(env)

#%% set random seed
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

    if args.method == 'COUPLE':
        nb_actions = 6
    elif args.method in ['ROTATION', 'AUTO']:
        nb_actions = 7
    elif args.method == 'NORMAL':
        nb_actions = env.action_space.shape[0]

#%% Set agent
    writer = SummaryWriter(log_dir='{}/tensorboardx_log'.format(args.output))
    agent = TD3(nb_states, nb_actions, args, noise=False)
    evaluate = Evaluator(args.validate_episodes,
                         args.validate_steps,
                         args.method, args.output,
                         episodes_length=args.max_episode_length
                         )

    # un-cite to load pretrained weight
    # agent.load_weights(args.resume)
    # agent.load_weights('bc_output/Walker2d-v2')

#%% train & test & behavior clone
    if args.mode == 'train':
        train(writer, args, agent, env, evaluate, MODE,
              debug=args.debug,
              num_interm=args.num_interm,
              visualize=False
              )
    elif args.mode == 'test':
        test(writer, args.validate_episodes, agent, env, args.window_length,
             evaluate, args.resume, visualize=True, debug=args.debug
             )
    elif args.mode == 'BC+FineTune':
        bc = behavior_clone(args, nb_states, nb_actions)
        bc.clone(agent, 100000)
    else:
        raise RuntimeError('undefined mode {}'.format(args.mode))
