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

        plt.ion()
        self.aim_his_pos = [self.aim.position]
        self.aim_his_v = [self.aim.velocity]
        self.asv_his_pos = [self.asv.position.data]
        self.asv_his_v = [self.asv.velocity.data]
        self.action_his = []

        self.observation_space = spaces.Box(low=0, high=50, shape=(12,))
        self.action_space = spaces.Box(low=-15, high=15, shape=(4,))
    
    def reset(self):
        """重设环境状态
            将目标点重置到(0, 0)之后，获取下一个目标点
            将船只重置为(0, 0)
        """
        self.aim.reset()
        self.aim.next_point(self.interval)
        self.asv.reset_state()
        aim_pos = self.aim.position
        aim_v = self.aim.velocity
        asv_pos = self.asv.position.data
        asv_v = self.asv.velocity.data
        self.aim_his_pos = [list(aim_pos)]
        self.aim_his_v = [list(aim_v)]
        self.asv_his_pos = [list(asv_pos)]
        self.asv_his_v = [list(asv_v)]
        self.action_his = []
        plt.ioff()
        plt.clf()
        plt.ion()
        return self.get_state()

    def get_state(self):
        """获取当前环境状态:asv6个状态 和 aim-asv5个相对状态 和 aim_theta"""
        asv_pos = self.asv.position.data
        asv_v = self.asv.velocity.data
        aim_pos = self.aim.position
        aim_v = self.aim.velocity
        
        delta_pos = aim_pos[0:2] - asv_pos[0:2]
        aim_theta = np.array([aim_pos[2]])
        delta_v = aim_v - asv_v

        state = np.concatenate((asv_pos, asv_v, delta_pos, aim_theta, delta_v), axis=0)
        return state

    def get_done(self):
        """对局结束：移动后的船不在目标点周围1m内"""
        if self.d_after_a > 1:
            return True
        return False
        
    def get_reward(self, action):
        # # V5 备用R
        # del_d = self.d_before_a - self.d_after_a
        # if del_d >= 0 and self.del_theta < math.pi/2:
        #     r1 = np.power(2, - 10 * self.d_after_a)
        # else:
        #     r1 = -1

        # r2 = np.power(math.e, - 5 * self.del_theta) - 1

        # # V3 R
        r1 = np.power(2, - 10 * self.d_after_a) - 1

        # 计算asv移动前后和此时aim距离的差
        del_d = self.d_before_a - self.d_after_a

        if del_d >= 0 and self.del_theta < math.pi/2:
            r2 = np.power(math.e, - 5 * self.del_theta)
        else:
            r2 = -1

        # r3 = 0
        # for i in self.del_action:
        #     r3 += 0.1 * (np.exp(-np.power(i, 2)/20) - 1)

        r =r1 + r2
        return r

    def get_reward_punish(self):
        return -10
        
    def step(self, action):
        # 注意因为reset中已经让aim移动，因此aim永远是asv要追逐的点
        aim_pos = self.aim.position[0:2]
        # 计算asv本步移动前和aim之间的距离
        asv_pos = self.asv.position.data[0:2]
        self.d_before_a = math.sqrt(np.sum(np.power((asv_pos - aim_pos), 2)))

        # 获得本次action和上次action的差
        self.del_action = action - self.asv.motor.data

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
        cur_aim_pos, cur_aim_v = self.aim.next_point(self.interval)
        # 此时aim已经是下一个要追逐的点，可以计算state
        state = self.get_state()

        # 记录坐标点及action，便于绘图
        self.aim_his_pos.append(list(cur_aim_pos))
        self.aim_his_v.append(list(cur_aim_v))
        self.asv_his_pos.append(list(cur_asv_pos.data))
        self.asv_his_v.append(list(cur_asv_v.data))
        self.action_his.append(list(action))

        return state, reward, done, ''

    def render(self):
        aim_his_pos = np.array(self.aim_his_pos)
        aim_his_v = np.array(self.aim_his_v)
        asv_his_pos = np.array(self.asv_his_pos)
        asv_his_v = np.array(self.asv_his_v)
        action_his = np.array(self.action_his)

        plt.clf()

        # 绘制轨迹图
        plt.subplot(2,2,1)
        # 绘制aim
        plt.plot(*zip(*aim_his_pos[:,[0,1]]), 'y', label='aim')
        # 绘制asv
        plt.plot(*zip(*asv_his_pos[:,[0,1]]), 'b', label='asv')
        plt.title('x-y')
        plt.legend()

        # 绘制action图
        plt.subplot(2,2,2)
        # my_x_ticks = np.arange(0, 30, 0.1)
        # plt.xticks(my_x_ticks)
        plt.plot(range(0, len(action_his)), action_his[:,0], label='a1')
        plt.plot(range(0, len(action_his)), action_his[:,1], label='a2')
        plt.plot(range(0, len(action_his)), action_his[:,2], label='a3')
        plt.plot(range(0, len(action_his)), action_his[:,3], label='a4')
        plt.title('action')
        plt.legend()

        # 绘制theta对比图
        plt.subplot(2,2,3)
        plt.plot(range(0, len(aim_his_pos)), aim_his_pos[:,2], label='aim')
        plt.plot(range(0, len(asv_his_pos)), asv_his_pos[:,2], label='asv')
        plt.title('theta')
        plt.legend()

        # 绘制asv的速度图
        plt.subplot(2,2,4)
        plt.plot(range(0, len(asv_his_v)), asv_his_v[:,0], label='u_asv')
        plt.plot(range(0, len(aim_his_v)), aim_his_v[:,0], label='u_aim')
        plt.plot(range(0, len(asv_his_v)), asv_his_v[:,1], label='v_asv')
        plt.plot(range(0, len(aim_his_v)), aim_his_v[:,1], label='v_aim')
        plt.plot(range(0, len(asv_his_v)), asv_his_v[:,2], label='r_asv')
        plt.plot(range(0, len(aim_his_v)), aim_his_v[:,2], label='r_aim')
        plt.title('u,v,r')
        plt.legend()

        plt.pause(0.1)