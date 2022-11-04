from collections import deque

import torch
import gym
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, input_num, output_num):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_num, 256)
        self.fc2 = nn.Linear(256, output_num)
        self.mse = nn.MSELoss()
        self.action_space = [i for i in range(output_num)]
        self.opt = optim.Adam(self.parameters(), lr=0.0001)
        self.output_num = output_num
        self.gamma = 1

    def forward(self, x):
        #x = torch.tensor(x).to(torch.float)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def train_model(self, batch):
        state = [i[0] for i in batch]
        action = [i[1] for i in batch]
        reward = [i[2] for i in batch]
        next_state = [i[3] for i in batch]
        mask = [i[4] for i in batch]

        state = torch.tensor(state).to(torch.float32)
        one_hot_action = np.zeros((len(action), self.output_num))
        for i, a in enumerate(action):
            one_hot_action[i][a] = 1
        action = torch.tensor(one_hot_action).float()
        #print('action ', action.shape)
        reward = torch.tensor(reward).to(torch.float32)
        next_state = torch.tensor(next_state).to(torch.float32)
        mask = torch.tensor(mask)

        pred = self(state)
        pred = torch.sum(pred.mul(action), dim=1)
        next_pred = self(next_state)
        next_pred = next_pred.max(1)[0]
        target = reward + mask * self.gamma * next_pred
        #print(pred, reward, next_pred, target)

        loss = F.mse_loss(pred, target.detach())
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def act(self, state, epsilon):
        if random.random() < epsilon:
            return random.choice(self.action_space)
        else:
            return np.argmax(self(torch.tensor(state).to(torch.float32)).detach().numpy())


def main():
    #game = "Acrobot-v1"
    game = 'CartPole-v0'
    #env = gym.envs.make("CartPole-v0")
    #env = gym.envs.make("MountainCar-v0")
    #env = gym.envs.make("Pendulum-v0")
    env = gym.envs.make(game)
    #env = gym.envs.make(game)
    batch_size = 2048
    mlen = 25000
    memory = deque(maxlen=mlen)
    agent = DQN(len(env.observation_space.sample()), env.action_space.n)
    episode = 30000
    epsilon = 1
    need_render = False
    reward_que = deque(maxlen=20)
    best_r = -10000

    for e in range(episode):
        done = False
        state, _ = env.reset()
        time = 0
        reward_sum = 0
        if e % 20 == 0:
            epsilon -= 0.1
            epsilon = max(epsilon, 0.01)
        while not done:
            action = agent.act(state, epsilon)
            next_state, reward, done, _, _ = env.step(action)

            reward_sum += reward

            mask = 0 if done else 1
            memory.append((state, action, reward, next_state, mask))
            if len(memory) == mlen:
                batch = random.sample(memory, batch_size)
                agent.train_model(batch)
            state = next_state
        reward_que.append(reward_sum)
        best_r = max(best_r, reward_sum)
        avg = sum(reward_que) / len(reward_que)
        #if avg >= 199:
        #    epsilon = 0
        #    need_render = True
        need_render = (0 < (e % 100) < 10)
        if need_render:
            env.set_render_mode('human')
        else:
            env.set_render_mode(None)
        if len(memory) == mlen and e % 10 == 0:
            print('Episode %d %d %f %d' % (e, avg, epsilon, best_r))


if __name__ == '__main__':
    main()

