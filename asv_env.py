#!/usr/bin/env python
# coding=utf-8

import numpy as np
import math
from asv_dynamic import ASV
from move_point import MovePoint
import matplotlib.pyplot as plt
import gym
from gym import spaces

class ASVEnv(gym.Env):
    """
    AUV 的环境
    AUV的状态(state)由六个部分组成，分别是：
        当前坐标x
        当前坐标y
        当前夹角theta
        转机1速度 u
        转机2速度 v
        转机3速度 r
    AUV的动作(action)是控制四个转机速度的列表[]
    AUV的状态变化通过调用c语言接口实现
    """
    def __init__(self, target_trajectory='linear', interval=0.1):
        self.target_trajectory = target_trajectory
        self.interval = interval
        if self.target_trajectory == 'linear':
            self.asv = ASV(math.pi/4, self.interval)
        else:
            self.asv = ASV(0, self.interval)
        self.aim = MovePoint(self.target_trajectory)
        # self.playground_shape = (-1, 7, -1, 7)

        plt.ion()
        self.aim_his = [self.aim.position[0:2]]
        self.asv_his = [self.asv.position.data[0:2]]
        self.action_his = []

        self.observation_space = spaces.Box(low=0, high=50, shape=(7,))
        self.action_space = spaces.Box(low=-15, high=15, shape=(4,))
    
    def reset(self):
        """重设环境状态
            将目标点重置到(0, 0)之后，获取下一个目标点
            将船只重置为(0, 0)
        """
        self.aim.reset()
        self.aim.next_point(self.interval)
        self.asv.reset_state()
        aim_pos = self.aim.position[0:2]
        asv_pos = self.asv.position.data[0:2]
        self.aim_his = [list(aim_pos)]
        self.asv_his = [list(asv_pos)]
        self.action_his = []
        plt.ioff()
        plt.clf()
        plt.ion()
        return self.get_state()

    def get_state(self):
        """获取当前环境状态"""
        asv_pos = self.asv.position.data[0:2]
        aim_pos = self.aim.position[0:2]
        theta = np.array([self.asv.position.theta, self.aim.position[2]])
        state = np.append(aim_pos - asv_pos, theta)
        state = np.append(state, self.asv.velocity.data)
        return state

    def get_done(self):
        """对局结束：移动后的船不在目标点周围1m内"""
        if self.d_after_a > 1:
            return True
        return False
        
    def get_reward(self, action):
        r1 = np.power(2, - 10 * self.d_after_a) - 1

        # 计算asv移动前后和此时aim距离的差
        del_d = self.d_before_a - self.d_after_a

        if del_d >= 0 and self.del_theta < math.pi/2:
            r2 = np.power(math.e, - 5 * self.del_theta)
        else:
            r2 = -1

        r = r1 + r2
        # a = np.sum(np.power(action, 2))
        # r2 = np.power(2, - a/100) - 1

        # r = r1 + (1/max(1,self.d+del_theta)) * r2
        return r

    def get_reward_punish(self):
        return -10
        
    def step(self, action):
        # 注意因为reset中已经让aim移动，因此aim永远是asv要追逐的点
        aim_pos = self.aim.position[0:2]
        # 计算asv本步移动前和aim之间的距离
        asv_pos = self.asv.position.data[0:2]
        self.d_before_a = math.sqrt(np.sum(np.power((asv_pos - aim_pos), 2)))

        # 在获得action之后，让asv根据action移动
        self.asv.motor = action
        # 让asv移动后，当前asv坐标更新为移动后的坐标
        cur_asv_pos, cur_asv_v = self.asv.move()

        # 计算asv移动后和aim之间的距离
        # 及移动后asv艏向角和此时目标航向的夹角 del_theta
        self.d_after_a = math.sqrt(np.sum(np.power((cur_asv_pos.data[0:2] - aim_pos), 2)))
        self.del_theta = abs(self.aim.position[2]-self.asv.position.theta) if abs(self.aim.position[2]-self.asv.position.theta) < math.pi\
            else math.pi * 2 - abs(self.aim.position[2]-self.asv.position.theta)
        
        # 奖励应该是对于当前aim，以及移动以后的asv计算
        done = self.get_done()
        # 注意奖励永远是根据当前aim坐标和当前asv坐标计算，当前aim尚未移动
        if done:
            reward = self.get_reward_punish()
        else:
            reward = self.get_reward(action)
        # 计算完奖励之后，可以移动aim坐标
        cur_aim = self.aim.next_point(self.interval)
        # 此时aim已经是下一个要追逐的点，可以计算state
        state = self.get_state()

        # 记录坐标点及action，便于绘图
        self.aim_his.append(list(cur_aim[0:2]))
        self.asv_his.append(list(cur_asv_pos.data[0:2]))
        self.action_his.append(list(action))

        return state, reward, done, ''

    def render(self):
        plt.clf()

        # 绘制轨迹图
        plt.subplot(1,2,1)
        # 绘制aim
        plt.plot(*zip(*self.aim_his), 'y', label='aim')
        # 绘制asv
        plt.plot(*zip(*self.asv_his), 'b', label='asv')
        plt.title('x-y')
        plt.legend()

        # 绘制action图
        plt.subplot(1,2,2)
        a = np.array(self.action_his)
        # my_x_ticks = np.arange(0, 30, 0.1)
        # plt.xticks(my_x_ticks)
        plt.plot(range(0, len(a)), a[:,0], label='a1')
        plt.plot(range(0, len(a)), a[:,1], label='a2')
        plt.plot(range(0, len(a)), a[:,2], label='a3')
        plt.plot(range(0, len(a)), a[:,3], label='a4')
        plt.title('action')
        plt.legend()

        plt.pause(0.1)