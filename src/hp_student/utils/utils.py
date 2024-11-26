import numpy as np
import pickle


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


