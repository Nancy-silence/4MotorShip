#!/usr/bin/env python
# coding=utf-8

import numpy as np
import math

class MovePoint(object):
    def __init__(self, target_trajectory='linear'):
        self.position = np.array([0.0, 0.0, 0.0])
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.t = 0.0
        self.target_trajectory = target_trajectory
        self.impl = getattr(self, target_trajectory)

    def reset(self):
        self.position = np.array([0.0, 0.0, 0.0])
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.t = 0.0

    # @property
    # def position(self):
    #     return self.position

    def next_point(self, interval):
        self.t += interval
        return self.impl(self.t)

    def trajectory_point(self, x):
        if self.target_trajectory == 'linear':
            return self.linear_trajectory(x)
        elif self.target_trajectory == 'func_sin':
            return self.func_sin_trajectory(x)
        else:
            return

    def linear(self, t):
        x = t / 5
        y = t / 5
        vx = 0.2
        vy = 0.2
        theta = math.atan2(vy, vx)
        u = math.sqrt(vx**2 + vy **2)
        v = 0.0
        r = 0.0
        self.position = np.array([x, y, theta])
        self.velocity = np.array([u, v, r])
        return self.position, self.velocity

    def linear_trajectory(self, x):
        if x >= 0:
            t = 5 * x
            y = t / 5
            vx = 0.2
            vy = 0.2
            theta = math.atan2(vy, vx)
            pos = np.array([x, y, theta])
        else:
            pos = np.array([0, 0, math.pi/4])
        return pos

    def func_sin(self, t):
        x = t / 5
        y = - np.cos(t/5) + 1
        vx = 0.2
        vy = 0.2 * math.sin(t/5)
        theta = math.atan2(vy, vx)
        u = math.sqrt(vx**2 + vy **2)
        v = 0.0
        r = (0.2*math.cos(t/5)) / (1 + np.power(math.sin(t/5),2))
        self.position = np.array([x, y, theta])
        self.velocity = np.array([u, v, r])
        return self.position , self.velocity

    def func_sin_trajectory(self, x):
        if x >= 0:
            t = 5 * x
            y = - np.cos(t/5) + 1
            vx = 0.2
            vy = 0.2 * math.sin(t/5)
            theta = math.atan2(vy, vx)
            pos = np.array([x, y, theta])
        else:
            pos = np.array([0, 0, 0])
        return pos

    def random(self, t):
        self.position = np.random.randint(0, 100, 2)
        return self.position, self.velocity