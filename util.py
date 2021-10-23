import os
import numpy as np
import torch


USE_CUDA = torch.cuda.is_available()
FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor


def print_green(prt):
    print("\033[92m {}\033[00m" .format(prt))


def print_yellow(prt):
    print("\033[93m {}\033[00m" .format(prt))


def to_numpy(var):
    if USE_CUDA:
        return var.cpu().data.numpy()
    else:
        var.data.numpy()


def to_tensor(array):
    return torch.autograd.Variable(
        torch.from_numpy(array),  requires_grad=False
    ).type(FLOAT)


def soft_update(target_net, source_net, tau):
    for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(
            (1.0 - tau) * target_param.data + tau * source_param.data
        )


def hard_update(target_net, source_net):
    for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(source_param.data)


def get_output_folder(root_dir, env_name):
    os.makedirs(root_dir, exist_ok=True)
    experiment_id = 0
    for folder_name in os.listdir(root_dir):
        if not os.path.isdir(os.path.join(root_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    root_dir = os.path.join(root_dir, env_name)
    root_dir = root_dir + '-run{}'.format(experiment_id)
    os.makedirs(root_dir, exist_ok=True)
    return root_dir


def duplicate_action(action):
    pairA = action[:3]
    pairB = action[3:]
    mask = np.array([-1, 1, 1])
    a = np.concatenate((pairA, pairB, pairA, pairB*mask, pairA*mask, pairB*mask))
    a = a/2
    return a



