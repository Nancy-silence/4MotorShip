#!/usr/bin/env python
# coding=utf-8

import numpy as np
import math

class MovePoint(object):
    def __init__(self, move_path='linear'):
        self.point = np.array([0.0, 0.0, 0.0])
        self.impl = getattr(self, move_path)

    def reset(self):
        self.point = np.array([0.0, 0.0, 0.0])

    @property
    def position(self):
        return self.point

    def next_point(self, interval):
        return self.impl(interval)

    def linear(self, interval):
        x, y, theta = self.point
        x += interval / 5
        y = x
        theta = math.pi / 4
        self.point = np.array([x, y, theta])
        return self.point

    def func_sin(self, interval):
        x, y, theta = self.point
        x += interval / 5
        y = 50 * np.sin(x / 10)
        theta = math.atan(5 * np.cos(x / 10))
        self.point = np.array([x, y, theta])
        return self.point

    def random(self, interval):
        self.point = np.random.randint(0, 100, 2)
        return self.point