import pickle
import numpy as np
from src.hp_student.utils.utils import type_of, shape


class ReplayMemory(object):
    """
    Replay memory class to store trajectories
    """

    def __init__(self, size, combined_experience_replay=False):
        """
        initializing the replay memory
        """
        self.combined_experience_replay = combined_experience_replay
        self.new_head = False
        self.k = 0
        self.head = -1
        self.full = False
        self.size = int(size)
        self.memory = None

    def initialize(self, experience):

        self.memory = [np.zeros(shape=(self.size, shape(exp)), dtype=type_of(exp)) for exp in experience]
        self.memory.append(np.zeros(shape=self.size, dtype=float))

    def add(self, experience):
        if self.memory is None:
            self.initialize(experience)
            print("initialized done")

        if len(experience) + 1 != len(self.memory):
            raise Exception('Experiment not the same size as memory', len(experience), '!=', len(self.memory))

        for e, mem in zip(experience, self.memory):
            mem[self.k] = e

        self.head = self.k
        self.new_head = True
        self.k += 1
        if self.k >= self.size:
            self.k = 0  # replace the oldest one with the latest one
            self.full = True

    def sample(self, batch_size):
        r = self.size
        if not self.full:
            r = self.k
        random_idx = np.random.choice(r, size=batch_size, replace=False)

        if self.combined_experience_replay:
            if self.new_head:
                random_idx[0] = self.head  # always add the latest one
                self.new_head = False

        return [mem[random_idx] for mem in self.memory]

    def get(self, start, length):
        return [mem[start:start + length] for mem in self.memory]

    def get_size(self):
        if self.full:
            return self.size
        return self.k

    def get_max_size(self):
        return self.size

    def reset(self):
        self.k = 0
        self.head = -1
        self.full = False
        self.memory = None
        self.new_head = False

    def shuffle(self):
        """
        to shuffle the whole memory
        """
        self.memory = self.sample(self.get_size())

    def save2file(self, file_path):
        with open(file_path, 'wb') as fp:
            pickle.dump(self.memory, fp)

    def load_memory_caches(self, path):

        with open(path, 'rb') as fp:
            memory = pickle.load(fp)
            if self.memory is None:
                self.memory = memory
            else:
                self.memory = np.hstack((self.memory, memory))

        print("Load memory caches, pre-filled replay memory!")
