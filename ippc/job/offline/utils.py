import numpy as np
import pickle
import pdb
import torch

DTYPE = torch.float
DEVICE = 'cuda'


def to_np(x):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return x


def to_torch(x, dtype=None, device=None):
    dtype = dtype or DTYPE
    device = device or DEVICE
    if type(x) is dict:
        return {k: to_torch(v, dtype, device) for k, v in x.items()}
    elif torch.is_tensor(x):
        return x.to(device).type(dtype)
    # import pdb; pdb.set_trace()
    return torch.tensor(x, dtype=dtype, device=device)


def to_device(x, device=DEVICE):
    if torch.is_tensor(x):
        return x.to(device)
    elif type(x) is dict:
        return {k: to_device(v, device) for k, v in x.items()}
    else:
        print(f'Unrecognized type in `to_device`: {type(x)}')
        pdb.set_trace()


# return [x.to(device) for x in xs]


def clip_or_wrap_func(a, a_min, a_max, clip_or_wrap):
    if clip_or_wrap == 0:
        return np.clip(a, a_min, a_max)
    return (a - a_min) % (a_max - a_min) + a_min


class ActionNoise:

    def __init__(self, action_dim, bounds, clip_or_wrap):
        self.action_dim = action_dim
        self.bounds = bounds
        self.clip_or_wrap = clip_or_wrap

    def sample(self) -> np.ndarray:
        pass

    def clip_or_wrap_action(self, action):
        if len(action) == 1:
            return clip_or_wrap_func(action, self.bounds[0], self.bounds[1], self.clip_or_wrap)
        return np.array([clip_or_wrap_func(a, self.bounds[0][k], self.bounds[1][k], self.clip_or_wrap[k]) for k, a in
                         enumerate(action)])

    def add_noise(self, action):
        sample = self.sample()
        action = self.clip_or_wrap_action(action + sample)
        return action


class OrnsteinUhlenbeckActionNoise(ActionNoise):

    def __init__(self, action_dim, bounds=(-1, 1), clip_or_wrap=0, mu=0, theta=0.15, sigma=0.1, dt=0.04):
        super().__init__(action_dim, bounds, clip_or_wrap)
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.X = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.X = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.X) * self.dt
        dx = dx + self.sigma * np.random.randn(len(self.X)) * np.sqrt(self.dt)
        self.X = self.X + dx
        return self.X


def shape(exp):
    if type(exp) is list:
        return len(exp)
    if type(exp) is np.ndarray:
        return len(exp)
    else:
        return 1


def type_of(exp):
    if type(exp) is bool:
        return bool
    else:
        return float


class ReplayMemory(object):
    """
    Replay memory class to store trajectories
    """

    def __init__(self, size, combined_experience_replay=False) -> object:
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
