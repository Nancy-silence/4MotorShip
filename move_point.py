#!/usr/bin/env python
# coding=utf-8

import numpy as np
import math

class MovePoint(object):
    def __init__(self, interval = 0.1, target_trajectory='linear', ud=0.3):
        self.interval = interval
        self.target_trajectory = target_trajectory
        self.ud = 0.3

        if self.target_trajectory == 'linear':
            self.position = np.array([1, 5, 0])
        elif self.target_trajectory == 'func_sin':
            self.position = np.array([0, 4, math.pi/4])
        else:
            #TODO:异常处理
            pass
        self.velocity = np.array([self.ud, 0, 0])

        self.aim_his_pos = [self.position]
        self.aim_his_v = [self.velocity]

        self.impl = getattr(self, target_trajectory)

    def reset(self):
        if self.target_trajectory == 'linear':
            self.position = np.array([1, 5, 0])
        elif self.target_trajectory == 'func_sin':
            self.position = np.array([0, 4, math.pi/4])
        else:
            #TODO:异常处理
            pass
        self.velocity = np.array([self.ud, 0, 0])

        self.aim_his_pos = [list(self.position)]
        self.aim_his_v = [list(self.velocity)]

    def observation(self):
        return self.position, self.velocity

    # @property
    # def position(self):
    #     return self.position

    def next_point(self):
        return self.impl()

    def trajectory_point(self, x):
        if self.target_trajectory == 'linear':
            return self.linear_trajectory(x)
        elif self.target_trajectory == 'func_sin':
            return self.func_sin_trajectory(x)
        else:
            return

    def linear(self):
        x = self.position[0] + self.ud * self.interval
        self.position = np.array([x, self.position[1], self.position[2]])

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
        x = self.position[0] + self.ud * self.interval / math.sqrt(1 + np.power(math.cos(self.position[0]),2))
        y = self.position[1] + self.ud * self.interval * math.cos(self.position[0]) / math.sqrt(1 + np.power(math.cos(self.position[0]),2))
        theta = math.atan2(math.cos(self.position[0]), 1)
        u = self.ud
        v = 0
        r = (theta - self.position[2]) / self.interval
        self.position = np.array([x, y, theta])
        self.velocity = np.array([u, v, r])

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
