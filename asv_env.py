#!/usr/bin/env python
# coding=utf-8

import numpy as np
import pandas as pd
import math
from asv_dynamic import ASV
from move_point import MovePoint
import matplotlib.pyplot as plt
import gym
from gym import spaces
import time

class ASVEnv(gym.Env):
    """
    ASV 的环境
    ASV的状态(state)由12个部分组成，分别是：
        asv的当前坐标x,当前坐标y,当前航角theta,前进方向速度 u,切向速度 v,转动速度 r
        asv和目标点间相对x,y,u,v,r
        目标点航角theta
    ASV的动作(action)是控制四个电机速度的列表[]
    ASV的状态变化通过调用c语言接口实现
    """
    def __init__(self, target_trajectory='linear', interval=0.1, ud=0.3, measure_bias=False):
        self.target_trajectory = target_trajectory
        self.interval = interval
        self.asv = ASV(self.interval, measure_bias)
        self.aim = MovePoint(self.interval, self.target_trajectory, ud)

        plt.ion()

        self.observation_space = spaces.Box(low=0, high=50, shape=(10,))
        self.action_space = spaces.Box(low=-6, high=6, shape=(4,))
    
    def reset(self):
        """重设环境状态
            将目标点重置到(0, 0)之后，获取下一个目标点
            将船只重置为(0, 0)
        """
        self.aim.reset()
        begin_pos, begin_v= self.aim.observation()
        self.asv.reset_state(begin_pos)
        self.aim.next_point()
    
        plt.ioff()
        plt.clf()
        plt.ion()
        return self.get_state()

    def get_state(self):
        """获取当前环境状态:asv6个状态 和 aim-asv5个相对状态 和 aim_theta"""
        aim_pos, aim_v = self.aim.observation()
        asv_pos, asv_v = self.asv.observation()
        aim_theta = aim_pos[2]
        asv_theta = asv_pos[2]
        
        delta_pos = aim_pos[0:2] - asv_pos[0:2]
        delta_theta = (0 - np.sign(aim_theta - asv_theta)) * (math.pi * 2 - abs(aim_theta - asv_theta)) if \
            abs(aim_theta - asv_theta) > math.pi else aim_theta - asv_theta
        delta_theta = np.array([delta_theta])
        delta_v = aim_v - asv_v

        state = np.concatenate((delta_pos, delta_theta, delta_v, self.asv.motor.data), axis=0)
        return state

    def get_done(self):
        """对局结束：移动后的船不在目标点周围1m内"""
        if self.d_after_a > 1:
            return True
        return False
        
    def get_reward(self, action):
        
        state = self.get_state()[0:6]
        error = np.sum(np.array([20,20,5,0.1,0.1,0.1]) * np.power(state,2))
        # 计算asv移动前后和此时aim距离的差
        del_d_target = self.d_before_a - self.d_after_a

        if del_d_target >= 0:
            r1 = np.exp(-error)
        else:
            r1 = -1
        

        sum_a = np.sum(np.power(action,2))
        r3 = 0.5 * (np.exp(-sum_a/100) - 1)

        motor_his = np.array(self.asv.asv_his_motor)
        a_nearby = motor_his[-min(40, len(motor_his)):,:]
        r4 = 0
        for i in range(4):
            std = np.nan_to_num(np.std(a_nearby[:,i], ddof=1))
            r4 += 0.25 * (np.exp(-std) - 1)

        r =r1 + r3 + r4
        # print(f'error:{error}, r1:{r1}, r3:{r3}, r4:{r4}, r:{r}')

        return r

    def get_reward_punish(self):
        return -25
        
    def step(self, action):
        # 注意因为reset中已经让aim移动，因此aim永远是asv要追逐的点
        aim_pos, aim_v= self.aim.observation()
         # 计算asv本步移动前和aim之间的距离
        asv_pos, asv_v= self.asv.observation()
        self.d_before_a = math.sqrt(np.sum(np.power((asv_pos[0:2] - aim_pos[0:2]), 2)))
        # 在获得action之后，让asv根据action移动
        self.asv.motor = action
        # 让asv移动后，当前asv坐标更新为移动后的坐标
        cur_asv_pos, cur_asv_v = self.asv.move()

        # 计算asv移动后和aim之间的距离
        self.d_after_a = math.sqrt(np.sum(np.power((cur_asv_pos[0:2] - aim_pos[0:2]), 2)))
        
        # 奖励应该是对于当前aim，以及移动以后的asv计算
        done = self.get_done()
        # 注意奖励永远是根据当前aim坐标和当前asv坐标计算，当前aim尚未移动
        if done:
            reward = self.get_reward_punish()
        else:
            reward = self.get_reward(action)
        # 计算完奖励之后，可以移动aim坐标
        cur_aim_pos, cur_aim_v = self.aim.next_point()
        # 此时aim已经是下一个要追逐的点，可以计算state
        state = self.get_state()

        return state, reward, done, ''

    def render(self):
        aim_his_pos = np.array(self.aim.aim_his_pos)
        aim_his_v = np.array(self.aim.aim_his_v)
        asv_his_pos = np.array(self.asv.asv_his_pos)
        asv_his_v = np.array(self.asv.asv_his_v)
        action_his = np.array(self.asv.asv_his_motor)

        plt.clf()

        # 绘制轨迹图
        plt.subplot(3,2,1)
        # 绘制aim
        plt.plot(*zip(*aim_his_pos[:,[0,1]]), 'y', label='aim')
        # 绘制asv
        plt.plot(*zip(*asv_his_pos[:,[0,1]]), 'b', label='asv')
        # my_ticks = np.arange(-4, 8, 0.5)
        # plt.yticks(my_ticks)
        plt.title('x-y')
        plt.legend()

        # 绘制误差ed图
        plt.subplot(3,2,2)
        aim_his_pos_fix = aim_his_pos[:-1]
        ed = np.sqrt(np.sum(np.power((asv_his_pos[:,[0,1]] - aim_his_pos_fix[:,[0,1]]), 2), axis=1))
        plt.plot(range(0, len(ed)), ed)
        plt.title('ed')

        # 绘制action图
        plt.subplot(3,2,3)
        plt.plot(range(0, len(action_his)), action_his[:,0], label='a1')
        plt.plot(range(0, len(action_his)), action_his[:,1], label='a2')
        plt.plot(range(0, len(action_his)), action_his[:,2], label='a3')
        plt.plot(range(0, len(action_his)), action_his[:,3], label='a4')
        my_y_ticks = np.arange(-6, 7, 1)
        plt.yticks(my_y_ticks)
        plt.title('action')
        plt.legend()

        # 绘制theta对比图
        plt.subplot(3,2,4)
        plt.plot(range(0, len(aim_his_pos)), aim_his_pos[:,2], label='aim')
        plt.plot(range(0, len(asv_his_pos)), asv_his_pos[:,2], label='asv')
        plt.title('theta')
        plt.legend()

        # 绘制asv的速度图
        plt.subplot(3,2,5)
        plt.plot(range(0, len(asv_his_v)), asv_his_v[:,0], label='u_asv')
        plt.plot(range(0, len(aim_his_v)), aim_his_v[:,0], label='u_aim')
        plt.plot(range(0, len(asv_his_v)), asv_his_v[:,1], label='v_asv')
        plt.plot(range(0, len(aim_his_v)), aim_his_v[:,1], label='v_aim')
        plt.plot(range(0, len(asv_his_v)), asv_his_v[:,2], label='r_asv')
        plt.plot(range(0, len(aim_his_v)), aim_his_v[:,2], label='r_aim')
        plt.title('u,v,r')
        plt.legend()

        plt.pause(0.1)

    def data_save_exl(self):
        aim_his_pos = np.array(self.aim.aim_his_pos)
        aim_his_v = np.array(self.aim.aim_his_v)
        asv_his_pos = np.array(self.asv.asv_his_pos)
        asv_his_v = np.array(self.asv.asv_his_v)
        action_his = np.array(self.asv.asv_his_motor)
        aim_s = np.hstack((aim_his_pos, aim_his_v))
        asv_s = np.hstack((asv_his_pos, asv_his_v))
        a = np.array(self.asv.asv_his_motor)

        aim_s_data = pd.DataFrame({'x':aim_s[:,0],'y':aim_s[:,1],'theta':aim_s[:,2],'u':aim_s[:,3],'v':aim_s[:,4],'r':aim_s[:,5]})
        asv_s_data = pd.DataFrame({'x':asv_s[:,0],'y':asv_s[:,1],'theta':asv_s[:,2],'u':asv_s[:,3],'v':asv_s[:,4],'r':asv_s[:,5]})
        a_data = pd.DataFrame({'f1':a[:,0],'f2':a[:,1],'f3':a[:,2],'f4':a[:,3]})

        writer = pd.ExcelWriter('State-Action.xlsx')
        aim_s_data.to_excel(writer, 'aim state', float_format='%.5f')
        asv_s_data.to_excel(writer, 'asv state', float_format='%.5f')
        a_data.to_excel(writer, 'action', float_format='%.5f')
        writer.save()
        writer.close()