import numpy as np


# Взят отсюда: https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
class OUStrategy:

    def __init__(self, action_space, mu=0, theta=0.15, max_sigma=0.3,
                 min_sigma=None, decay_period=100000):
        if min_sigma is None:
            min_sigma = max_sigma
        self.mu = mu
        # mu表示长期均值
        self.theta = theta
        # 正反馈参数，控制噪声值回归到长期均值mu的速度
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        # sigma是噪声的标准差，控制噪声的基线值

        self.min_sigma = min_sigma
        self.decay_period = decay_period
        # 噪声标准差从max-sigma到min-sigma的时间步数
        self.dim = np.prod(action_space.low.shape)
        # 用于计算动作空间的维度
        self.low = action_space.low
        self.high = action_space.high
        # 获取动作空间的取值范围
        self.state = np.ones(self.dim) * self.mu
        # 初始化状态

    def reset(self):
        self.state = np.ones(self.dim) * self.mu
        # 初始化函数

    def evolve_state(self):
        # 用于更新OU过程的状态
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

    def get_action_from_raw_action(self, action, t=0):
        # 用于获取带有噪声的动作
        ou_state = self.evolve_state()
        self.sigma = (self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t * 1.0 / self.decay_period))
        # 计算噪声标准差，会随着时间增加而衰减
        return np.clip(action + ou_state, self.low, self.high)
    # 计算带有噪声的动作，将其裁减到动作空间的取值范围内