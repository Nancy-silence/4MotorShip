#!/usr/bin/env python
# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
CUDA = torch.cuda.is_available()

class ASVActorNet(nn.Module):
    def __init__(self, n_states, n_actions, n_neurons=300, a_bound=1):
        super().__init__()
        self.bound = a_bound

        self.fc1 = nn.Linear(n_states, n_neurons)
        self.fc1.weight.data.normal_(0, 0.1)
        torch.nn.init.uniform_(self.fc1.bias.data, 0, 0.1)

        self.fc2 = nn.Linear(n_neurons, 150)
        self.fc2.weight.data.normal_(0, 0.1)
        torch.nn.init.uniform_(self.fc2.bias.data, 0, 0.1)

        self.out = nn.Linear(150, n_actions)
        torch.nn.init.xavier_uniform_(self.out.weight.data, gain=1)
        torch.nn.init.uniform_(self.out.bias.data, 0, 0.5)
        if CUDA:
            self.bound = torch.FloatTensor([self.bound]).cuda()
        else:
            self.bound = torch.FloatTensor([self.bound])

    def forward(self, x):
        """
        定义网络结构: 隐藏层1(100)->ReLU激活->隐藏层2(50)->ReLU激活->输出层->tanh激活->*bound 输出
        """
        x = x.cuda() if CUDA else x
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.out(x)
        action_value = torch.tanh(x)
        action_value = action_value * self.bound
        return action_value


class ASVCriticNet(nn.Module):
    def __init__(self, n_states, n_actions, n_neurons=300, a_bound=1):
        super().__init__()

        self.fc1 = nn.Linear(n_states+n_actions, n_neurons)
        self.fc1.weight.data.normal_(0, 0.1)

        self.fc2 = nn.Linear(n_neurons, n_neurons)
        self.fc2.weight.data.normal_(0, 0.1)

        self.out = nn.Linear(n_neurons, 1)
        # self.out.weight.data.normal_(0, 0.1)

    def forward(self, s, a):
        """
            定义网络结构: 隐藏层1(64)->ReLU激活->隐藏层2(32)->ReLU激活->输出层输出
        """
        x = torch.cat((s, a), dim=-1)
        x = x.cuda() if CUDA else x
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.out(x)
        # q_value = torch.tanh(x)
        return x