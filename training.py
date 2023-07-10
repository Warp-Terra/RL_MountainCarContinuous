import torch
import torch.optim as optim
import torch.nn.functional as F
import gym
import numpy as np
import copy
import random
from model import Actor
from model import Critic
from buffer import Buffer
from noise import OUStrategy
from collections import deque

GAMMA = 0.99
TAU = 1e-3
# 用于实现目标网络参数和actor-critic参数之间的混合更新。这个表明更新速度很慢，有利于稳定
BUF_SIZE = 4096
# 经验储存区大小
BATCH_SIZE = 256
# 每次从经验回放区取样训练的大小
LR = 1e-4
# 优化器学习率，用于决定模型参数在每次训练迭代中的更新幅度。更新幅度小，训练稳定

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class DDPG:

    def __init__(self, state_dim, action_dim):
        self.critic = Critic(state_dim, action_dim).to(device)
        # 创建一个critic，输入状态维度和行为维度，放在device上运行
        self.target_c = copy.deepcopy(self.critic)
        # 创建一个critic的深度拷贝，用于计算目标值，有助于训练的稳定性

        self.actor = Actor(state_dim).to(device)
        # 创建一个actor，只输入状态维度，放在device上运行
        self.target_a = copy.deepcopy(self.actor)
        # 创建一个actor的深度拷贝，用于提供稳定的目标

        self.optimizer_c = optim.Adam(self.critic.parameters(), lr=LR)
        self.optimizer_a = optim.Adam(self.actor.parameters(), lr=LR)
        # 创建Adam优化器，用于更新critic和actor的参数

    def act(self, state):
        state = torch.from_numpy(np.array(state)).float().to(device)
        # 将state转化为numpy数组，然后又将其转换为Pytorch张量，再将其转换为浮点型，最后送入设备
        return self.actor.forward(state).detach().squeeze(0).cpu().numpy()
    # 调用forward的方法，输入state，使用detach阻断反向转播，用squeeze来压缩维度，将设备转到CPU上，最后又转换为数组

    def update(self, batch):
        # 一开始，接受一个batch
        states, actions, rewards, next_states, dones = zip(*batch)
        # 将batch中的数据解压缩
        states = torch.from_numpy(np.array(states)).float().to(device)
        actions = torch.from_numpy(np.array(actions)).float().to(device)
        rewards = torch.from_numpy(np.array(rewards)).float().to(device).unsqueeze(1)
        next_states = torch.from_numpy(np.array(next_states)).float().to(device)
        dones = torch.from_numpy(np.array(dones)).to(device)
        # 分别提取数据，并将其转换为张量，并移到设备中去（GPU），再rewards中增加一个新的维度，为了方便后续的计算

        Q_current = self.critic(states, actions)
        # 计算当前状态和行为的Q值
        Q_next = self.target_c(next_states, self.target_a(next_states).detach())
        # 计算下一个状态和行为的Q值，并且使用detach来阻隔反向转播
        y = (rewards + GAMMA * Q_next).detach()
        # 希望Q能接近的值

        ##################Update critic#######################
        loss_c = F.mse_loss(y, Q_current)
        self.optimizer_c.zero_grad()
        loss_c.backward()
        self.optimizer_c.step()
        # 用损失函数计算Q与y之间的误差，然后清零优化器的梯度，在反向传播误差，然后更新critic的参数

        ##################Update actor#######################
        loss_a = -self.critic.forward(states, self.actor(states)).mean()
        self.optimizer_a.zero_grad()
        loss_a.backward()
        self.optimizer_a.step()
        # 计算actor网络的损失函数，清零梯度，反向传播误差，并更新actor的网络参数

        ##################Update targets#######################
        for target_pr, pr in zip(self.target_a.parameters(), self.actor.parameters()):
            target_pr.data.copy_(TAU * pr.data + (1 - TAU) * target_pr.data)

        for target_pr, pr in zip(self.target_c.parameters(), self.critic.parameters()):
            target_pr.data.copy_(TAU * pr.data + (1 - TAU) * target_pr.data)
            # 使用线性插值的方法进行更新（软更新），以稳定学习过程。TAU是介于0-1的参数，用于控制更新的速度，以保持稳定
        
    def testing(self, num_repeat=100):
        env_ = gym.make('MountainCarContinuous-v0')
        rews = np.zeros(shape=(num_repeat, ))
        for k in range(num_repeat):
            state = env_.reset()
            done = False
            while not done:
                action = self.act(state)
                state, reward, done, _ = env_.step([action])
                rews[k] += reward
        return np.mean(rews)
    # 用于测试
    
episodes = 10000

seed = 22
env = gym.make('MountainCarContinuous-v0')
env.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
# 为选取22个种子，为以上的环境添加种子

agent = DDPG(2, 1)
buf = Buffer(BUF_SIZE)
noise = OUStrategy(env.action_space, min_sigma=1e-4)
updates_noise = 0
for episode in range(episodes):
    state = env.reset()
    episode_reward = 0
    done = False
    total_reward = 0
    while not done:
        action = agent.act(state)
        action = noise.get_action_from_raw_action(action, updates_noise)
        # 加上噪声
        updates_noise += 1
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        buf.add((state, action, reward, next_state, done))
        if len(buf) >= BATCH_SIZE:
            agent.update(buf.sample(BATCH_SIZE))
        state = next_state
    print(f"I did {episode}th episode. Result: {total_reward}, sigma = {noise.sigma}")
    if not episode % 10:
        print(f'test_mean_reward = {agent.testing()}')

torch.save(agent.actor.state_dict(), 'actor_model.pth')
# torch.save(agent.critic.state_dict(), 'critic_model.pth')