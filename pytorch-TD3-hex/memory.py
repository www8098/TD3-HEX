import collections
from util import *
import torch
import warnings
import copy
import random
import numpy as np

# [reference] https://github.com/matthiasplappert/keras-rl/blob/master/rl/memory.py

# This is to be understood as a transition: Given `current_state`, performing `ActionNoise`
# yields `reward` and results in `next_state`, which might be `terminal`.
Experience = collections.namedtuple('Experience', 'current_state, act, reward, next_state, terminal1')


def sample_batch_indexes(low, high, size):
    if high - low >= size:
        b_ids = random.sample(range(low, high), size)
    else:
        # Not enough data. Help ourselves with sampling from the range, but the same index
        # can occur multiple times. This is not good and should be avoided by picking a
        # large enough warm-up phase.
        warnings.warn('needs more warm up steps')
        b_ids = np.random.random_integers(low, high - 1, size=size)
    assert len(b_ids) == size
    return b_ids


class RingBuffer(object):
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = [None for _ in range(maxlen)]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if index < 0 or index >= self.length:
            raise KeyError()
        return self.data[(self.start + index) % self.maxlen]

    def append(self, v):
        if self.length < self.maxlen:
            self.length += 1
        elif self.length == self.maxlen:
            self.start = (self.start + 1) % self.maxlen
        else:
            raise RuntimeError()
        self.data[(self.start + self.length - 1) % self.maxlen] = v


def zero_observ(ob):
    if hasattr(ob, 'shape'):  # 判断是否包含 shape 属性
        return np.zeros(ob.shape)
    elif hasattr(ob, '__iter__'):  # 可迭代？
        out = list()
        for i in ob:
            out.append(zero_observ(i))
        return out
    else:
        return 0.


class BasicMemory(object):
    def __init__(self, window_length, ignore_episode_boundaries=False):
        self.window_length = window_length
        self.ignore_episode_boundaries = ignore_episode_boundaries

        self.recent_observations = collections.deque(maxlen=window_length)
        self.recent_terminals = collections.deque(maxlen=window_length)

    def append(self, ob, terminal, training=True):
        self.recent_terminals.append(terminal)
        self.recent_observations.append(ob)


# BasicMemory 中维护 recent_observations and  recent_terminals
class SequentialMemory(BasicMemory):
    def __init__(self, limit, **kwargs):
        super(SequentialMemory, self).__init__(
            **kwargs)  # initialize kwargs as a dict，  store window_length and ignore_episode_boundaries

        self.limit = limit
        self.actions = RingBuffer(limit)  # actions rewards terminals observations ring buffers
        self.rewards = RingBuffer(limit)  # maxlen == limit
        self.terminals = RingBuffer(limit)
        self.observations = RingBuffer(limit)

    def sample(self, bsize, b_ids=None):  # 返回experience存储s0 s1 ActionNoise reward terminal， state为一串observation序列
        if b_ids is None:
            b_ids = sample_batch_indexes(0, len(self.observations) - 1, size=bsize)  # 从observation里随机抽取batch_size个idxs
        b_ids = np.array(b_ids) + 1
        assert np.min(b_ids) >= 1
        assert np.max(b_ids) < len(self.observations)
        assert len(b_ids) == bsize

        # Create experiences
        experiences = list()
        for index in b_ids:
            terminal0 = self.terminals[index - 2] if index >= 2 else False
            while terminal0:
                index = sample_batch_indexes(1, len(self.observations), size=1)[
                    0]  # nb_entries == len(self.observations) == 当前长度 ??
                terminal0 = self.terminals[index - 2] if index >= 2 else False  # 随机抽取idx，直到上一步 terminal 为 false
            assert 1 <= index < len(self.observations)

            # This code is slightly complicated by the fact that subsequent observations might be
            # from different episodes. We ensure that an experience never spans multiple episodes.
            # This is probably not that important in practice but it seems cleaner.
            current_state = [self.observations[index - 1]]
            for offset in range(0, self.window_length - 1):
                current_idx = index - 2 - offset  # 上一步observation 对应s0
                current_terminal = self.terminals[
                    current_idx - 1] if current_idx - 1 > 0 else False  # 用于判断是否完成一个 trajectory ？
                if current_idx < 0 or (not self.ignore_episode_boundaries and current_terminal):
                    # The previously handled ob was terminal, don't add the current one.
                    # Otherwise we would leak into a different episode.
                    break
                current_state.insert(0, self.observations[current_idx])  # 导入 index 之前 window_length 内的 ob
            while len(current_state) < self.window_length:
                current_state.insert(0, zero_observ(current_state[0]))  # 用0填充 功能和Memory中的get_recent_state类似

            # Okay, now we need to create the follow-up state. This is current_state shifted on timestep
            # to the right. Again, we need to be careful to not include an ob from the next
            # episode if the last state is terminal.
            next_state = [np.copy(x) for x in current_state[1:]]  # 复制state0[1:], next_state 比 current_state 滞后一个observation
            next_state.append(self.observations[index])

            assert len(current_state) == self.window_length
            assert len(next_state) == len(current_state)
            experiences.append(Experience(current_state=current_state, act=self.actions[index - 1],
                                          reward=self.rewards[index - 1],
                                          next_state=next_state,
                                          terminal1=self.terminals[index - 1]))
        assert len(experiences) == bsize
        return experiences

    def sample_and_split(self, bsize, b_ids=None):  # 拆分sample获得的experience， 分别写入buffer
        experiences = self.sample(bsize, b_ids)
        state0_batch = list()
        reward_batch = list()
        action_batch = list()
        terminal1_batch = list()
        state1_batch = list()
        for experience in experiences:
            state0_batch.append(experience.current_state)
            state1_batch.append(experience.next_state)
            reward_batch.append(experience.reward)
            action_batch.append(experience.act)
            if experience.terminal1:
                terminal1_batch.append(0.)
            else:
                terminal1_batch.append(1.)  # 0代表terminal为true， 为false时写入1

        # Prepare and validate parameters.
        state0_batch = np.array(state0_batch).reshape(bsize, -1)
        state1_batch = np.array(state1_batch).reshape(bsize, -1)
        terminal1_batch = np.array(terminal1_batch).reshape(bsize, -1)
        reward_batch = np.array(reward_batch).reshape(bsize, -1)
        action_batch = np.array(action_batch).reshape(bsize, -1)
        return state0_batch, action_batch, reward_batch, state1_batch, terminal1_batch

    def append(self, ob, act, reward, terminal, training=True):
        super(SequentialMemory, self).append(ob, terminal, training=training)  # 更新recent_observations

        if training:
            self.observations.append(ob)  # 环形缓冲定义的append， 会改变start保证数据的顺序性
            self.actions.append(act)
            self.rewards.append(reward)
            self.terminals.append(terminal)
