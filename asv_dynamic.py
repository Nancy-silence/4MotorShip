#!/usr/bin/env python
# coding=utf-8

import numpy as np
import math

class Dim3Position(object):
    def __init__(self):
        self.pos = np.array([0.0, 0.0, 0.0])

    @property
    def x(self):
        return self.pos[0]

    @x.setter
    def x(self, nx):
        self.pos[0] = nx

    @property
    def y(self):
        return self.pos[1]

    @y.setter
    def y(self, ny):
        self.pos[1] = ny

    @property
    def theta(self):
        return self.pos[2]

    @theta.setter
    def theta(self, ntheta):
        self.pos[2] = ntheta

    @property
    def data(self):
        return self.pos.copy()

    def __str__(self):
        return f'x: {self.x}, y: {self.y}, theta: {self.theta}'

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.theta == other.theta

class Dim3Velocity(object):
    def __init__(self):
        self.velocity = np.array([0.0, 0.0, 0.0])

    @property
    def u(self):
        return self.velocity[0]

    @u.setter
    def u(self, nu):
        self.velocity[0] = nu

    @property
    def v(self):
        return self.velocity[1]

    @v.setter
    def v(self, nv):
        self.velocity[1] = nv

    @property
    def r(self):
        return self.velocity[2]

    @r.setter
    def r(self, nr):
        self.velocity[2] = nr

    @property
    def data(self):
        return self.velocity.copy()

    def __str__(self):
        return f'u: {self.u}, v: {self.v}, r: {self.r}'

    def __eq__(self, other):
        return self.u == other.u and self.v == other.v and self.r == other.r

class Dim4Motor(object):
    def __init__(self):
        self.motor = np.array([0.0, 0.0, 0.0, 0.0])

    @property
    def a1(self):
        return self.motor[0]

    @a1.setter
    def a1(self, na1):
        self.motor[0] = na1
    
    @property
    def a2(self):
        return self.motor[1]

    @a2.setter
    def a2(self, na2):
        self.motor[1] = na2

    @property
    def a3(self):
        return self.motor[2]

    @a3.setter
    def a3(self, na3):
        self.motor[2] = na3

    @property
    def a4(self):
        return self.motor[3]

    @a4.setter
    def a4(self, na4):
        self.motor[3] = na4

    @property
    def data(self):
        return self.motor.copy()

    def __str__(self):
        return f'a1: {self.a1}, a2: {self.a2}, a3: {self.a3}, a4: {self.a4}'

class ASV(object):

    def __init__(self, time_interval = 0.1, measure_bias = False):
        """
        @param: time_interval控制系统决策时间
        """
        self.time_interval = time_interval
        self.measure_bias = measure_bias
        self.__position = Dim3Position()
        self.__position_bias = Dim3Position()
        self.__velocity = Dim3Velocity()
        self.__motor = Dim4Motor()

        self.asv_his_pos = [self.__position.data]
        self.asv_his_v = [self.__velocity.data]
        self.asv_his_motor = [self.__motor.data]
    
    @property
    def position(self):
        return self.__position

    @property
    def velocity(self):
        return self.__velocity

    @property
    def position_bias(self):
        return self.__position_bias

    @property
    def motor(self):
        return self.__motor
    
    @motor.setter
    def motor(self, motor):
        self.__motor.a1, self.__motor.a2, self.__motor.a3, self.__motor.a4 = motor 

    def reset_state(self, begin_pos):
        self.__position.x = begin_pos[0] # + np.random.rand() - 0.5
        self.__position.y = begin_pos[1] # + np.random.rand() - 0.5
        self.__position.theta = begin_pos[2] # + np.random.uniform(-math.pi/2, math.pi/2)
        self.__velocity.u, self.__velocity.v, self.__velocity.r = 0, 0, 0
        self.motor = (0, 0, 0, 0)

        self.asv_his_pos = [list(self.__position.data)]
        self.asv_his_v = [list(self.__velocity.data)]
        self.asv_his_motor = []

        if self.measure_bias:
            self.obs_add_bias()
        return self.observation()

    def move(self):
        from c_env.step import step
        obs = np.append(self.position.data, self.velocity.data)
        next_obs = step(obs, self.motor.data, self.time_interval)
        self.__position.x, self.__position.y, self.__position.theta, \
            self.__velocity.u, self.__velocity.v, self.__velocity.r = next_obs

        self.asv_his_pos.append(list(self.__position.data))
        self.asv_his_v.append(list(self.__velocity.data))
        self.asv_his_motor.append(list(self.__motor.data))

        if self.measure_bias:
            self.obs_add_bias()
        return self.observation()

    def obs_add_bias(self):
        # self.__position_bias.x = self.__position.x + (np.random.rand()-0.5)/50
        # self.__position_bias.y = self.__position.y + (np.random.rand()-0.5)/50
        self.__position_bias.x = self.__position.x + np.random.normal(0, 0.02)
        self.__position_bias.y = self.__position.y + np.random.normal(0, 0.02)
        self.__position_bias.theta = self.__position.theta

    def observation(self):
        if self.measure_bias:
            return self.__position_bias.data, self.__velocity.data
        else:
            return self.__position.data, self.__velocity.data
    

    
    

# if __name__ == '__main__':
#     ship = ASV()
#     ship.reset_state()
#     print(ship.position.data,ship.velocity.data,ship.motor.data)
#     ship.motor = np.array([2,1,-1,-2])
#     ship.move()
#     print(ship.position.data,ship.velocity.data,ship.motor.data)