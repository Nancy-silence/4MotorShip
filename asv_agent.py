#!/usr/bin/env python
# coding=utf-8

import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from asv_nn import ASVActorNet, ASVCriticNet
from Utils import ExpReplay, soft_update, NormalActionNoise, OUActionNoise
CUDA = torch.cuda.is_available()

class DDPG(object):
    def __init__(self, n_states, n_actions, a_bound=1, lr_a=0.0001, lr_c=0.001, tau=0.01, gamma=0.9,
                 MAX_MEM=10000, MIN_MEM=None, BATCH_SIZE=128, noise_type='OU', train=True, **kwargs):
        # 参数复制
        self.n_states, self.n_actions = n_states, n_actions
        self.bound = a_bound
        # 创建神经网络并指定优化器
        self._build_net()
        # 开启cuda
        if CUDA:
            self.cuda()

        if train:
            # 参数复制
            self.tau, self.gamma = tau, gamma
            self.batch_size = BATCH_SIZE
            # 记录agent跑的step数
            self.run_step = 0
            # 初始化训练指示符
            self.start_train = False
            self.mem_size = 0
            # 创建经验回放池
            self.memory = ExpReplay(n_states, n_actions, exp_size=MAX_MEM, exp_thres=MIN_MEM)  # s, a, r, d, s_
            # 指定噪声类型
            self.noise_type = noise_type
            # 指定summary writer
            if 'summary_path' in kwargs:
                self._build_summary_writer(kwargs['summary_path'])
            else:
                self._build_summary_writer()
            # 指定优化器
            self.actor_optim = torch.optim.Adam(self.actor_eval.parameters(), lr=lr_a)
            self.critic_optim = torch.optim.Adam(self.critic_eval.parameters(), lr=lr_c)
            # 约定损失函数
            self.mse_loss = nn.MSELoss()
        

    def _build_net(self):
        n_states, n_actions = self.n_states, self.n_actions
        self.actor_eval = ASVActorNet(n_states, n_actions)
        self.actor_target = ASVActorNet(n_states, n_actions)
        self.critic_eval = ASVCriticNet(n_states, n_actions)
        self.critic_target = ASVCriticNet(n_states, n_actions)
        self.actor_target.load_state_dict(self.actor_eval.state_dict())
        self.critic_target.load_state_dict(self.critic_eval.state_dict())
        # print(self.actor_eval.state_dict())
        # print(self.critic_eval.state_dict())

    def build_noise(self, mu, sigma):
        if self.noise_type == 'OU':
            self.noise = OUActionNoise(mu * np.ones(self.n_actions), sigma * np.ones(self.n_actions))
        else:  # self.noise_type == 'Normal'
            self.noise = NormalActionNoise(mu * np.ones(self.n_actions), sigma * np.ones(self.n_actions))

    def _build_summary_writer(self, summary_path=None):
        if summary_path:
            self.summary_writer = SummaryWriter(log_dir=summary_path)
        else:
            self.summary_writer = SummaryWriter()

    def get_summary_writer(self):
        return self.summary_writer

    def get_action(self, s):
        s = torch.unsqueeze(torch.FloatTensor(s), 0)
        action = self.actor_eval.forward(s).detach().cpu().numpy()
        self.run_step += 1
        return action[0]

    def get_action_noise(self, state):
        action = self.get_action(state)
        action_noise = self.noise()
        # print(action, action_noise)
        action += action_noise
        action = np.clip(action, -self.bound, self.bound)
        self.run_step += 1
        return action[0]

    def learn(self):
        """训练网络"""
        # 将eval网络参数赋给target网络
        soft_update(self.actor_target, self.actor_eval, self.tau)
        soft_update(self.critic_target, self.critic_eval, self.tau)

        # 获取batch并拆解
        batch = self.memory.get_batch_splited_tensor(CUDA, self.batch_size)
        if batch is None:
            return None, None
        else:
            self.start_train = True
        batch_cur_states, batch_actions, batch_rewards, batch_dones, batch_next_states = batch
        # 计算target_q，指导cirtic更新
        # 通过a_target和next_state计算target网络会选择的下一动作 next_action；通过target_q和next_states、刚刚计算的next_actions计算下一状态的q_values
        target_q_next = self.critic_target(batch_next_states, self.actor_target(batch_next_states))
        target_q = batch_rewards + self.gamma * (1 - batch_dones) * target_q_next  # 如果done，则不考虑未来
        # 指导critic更新
        q_value = self.critic_eval(batch_cur_states, batch_actions)
        td_error = self.mse_loss(target_q, q_value)
        self.critic_optim.zero_grad()
        td_error.backward()
        self.critic_optim.step()

        # 指导actor更新
        policy_loss = self.critic_eval(batch_cur_states, self.actor_eval(batch_cur_states))  # 用更新的eval网络评估这个动作
        # 如果 a是一个正确的行为的话，那么它的policy_loss应该更贴近0
        loss_a = -torch.mean(policy_loss)
        self.actor_optim.zero_grad()
        loss_a.backward()
        self.actor_optim.step()
        return td_error.detach().cpu().numpy(), loss_a.detach().cpu().numpy()

    def learn_batch(self):
        if 'learn_step' not in self.__dict__:
            self.learn_step = 0
        c_loss, a_loss = self.learn()
        if c_loss is not None:
            self.summary_writer.add_scalar('c_loss', c_loss, self.learn_step)
            self.summary_writer.add_scalar('a_loss', a_loss, self.learn_step)
            self.learn_step += 1

    def add_step(self, s, a, r, d, s_):
        self.memory.add_step(s, a, r, d, s_)
        self.mem_size += 1

    def save(self, episode, target_trajectory):
        state = {
            'actor_eval_net': self.actor_eval.state_dict(),
            'actor_target_net': self.actor_target.state_dict(),
            'critic_eval_net': self.critic_eval.state_dict(),
            'critic_target_net': self.critic_target.state_dict(),
            'episode': episode,
            'learn_step': self.learn_step,
            'run_step' : self.run_step
        }
        torch.save(state, f'./model/{target_trajectory}.pth')

    def load(self, model_path):
        if not os.path.exists(model_path):
            return 0
        saved_state = torch.load(model_path, map_location=torch.device('cpu'))
        self.actor_eval.load_state_dict(saved_state['actor_eval_net'])
        self.actor_target.load_state_dict(saved_state['actor_target_net'])
        self.critic_eval.load_state_dict(saved_state['critic_eval_net'])
        self.critic_target.load_state_dict(saved_state['critic_target_net'])
        self.learn_step = saved_state['learn_step']
        self.run_step = saved_state['run_step']
        return saved_state['episode'] + 1

    def cuda(self):
        self.actor_eval.cuda()
        self.actor_target.cuda()
        self.critic_eval.cuda()
        self.critic_target.cuda()



