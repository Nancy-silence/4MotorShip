#!/usr/bin/env python
# coding=utf-8

import numpy as np
import math

class MovePoint(object):
    def __init__(self, interval = 0.1, target_trajectory='linear', ud=0.3):
        self.interval = interval
        self.target_trajectory = target_trajectory
        self.ud = ud

        if self.target_trajectory == 'linear':
            self.position_iter = np.array([0, 0, 0])
        elif self.target_trajectory == 'func_sin':
            self.position_iter = np.array([0, 0, math.pi/4])
        else:
            #TODO:异常处理
            pass
        self.velocity_iter = np.array([self.ud, 0, 0])

        self.impl = getattr(self, target_trajectory)

    def reset(self, pos):
        self.start_pos = pos      # ASV起始位置（x,y,fai)
        self.start_pos[2] = 0     # 目标轨迹起始点设置为(x,y,0)
        self.position = self.start_pos + self.position_iter
        self.velocity = self.velocity_iter

        self.aim_his_pos = [list(self.position)]
        self.aim_his_v = [list(self.velocity)]
        return self.position , self.velocity

    def observation(self):
        return self.position, self.velocity

    def next_point(self):
        return self.impl()
    
    def last_point(self):
        return self.aim_his_pos[-2], self.aim_his_v[-2]

    def trajectory_point(self, x):
        if self.target_trajectory == 'linear':
            return self.linear_trajectory(x)
        elif self.target_trajectory == 'func_sin':
            return self.func_sin_trajectory(x)
        else:
            return

    def linear(self):
        x = self.position_iter[0] + self.ud * self.interval
        self.position_iter = np.array([x, self.position_iter[1], self.position_iter[2]])
       
        self.position = self.start_pos + self.position_iter
        self.velocity = self.velocity_iter
        self.aim_his_pos.append(list(self.position))
        self.aim_his_v.append(list(self.velocity))
        return self.position, self.velocity

    def linear_trajectory(self, x):
        if x >= 1:
            pos = np.array([x, 5, 0])
        else:
            pos = np.array([1, 5, 0])
        return pos

    def func_sin(self):
        x = self.position_iter[0] + self.ud * self.interval / math.sqrt(1 + np.power(math.cos(self.position_iter[0]),2))
        y = self.position_iter[1] + self.ud * self.interval * math.cos(self.position_iter[0]) / math.sqrt(1 + np.power(math.cos(self.position_iter[0]),2))
        theta = math.atan2(math.cos(x), 1)
        u = self.ud
        v = 0
        r = (theta - self.position_iter[2]) / self.interval
        self.position_iter = np.array([x, y, theta])
        self.velocity_iter = np.array([u, v, r])

        self.position = self.start_pos + self.position_iter
        self.velocity = self.velocity_iter
        self.aim_his_pos.append(list(self.position))
        self.aim_his_v.append(list(self.velocity))
        return self.position , self.velocity

    def func_sin_trajectory(self, x):
        if x >= 0:
            y = math.sin(x) + 4
            theta = math.atan2(math.cos(x), 1)
            pos = np.array([x, y, theta])
        else:
            pos = np.array([0, 4, math.pi/4])
        return pos
