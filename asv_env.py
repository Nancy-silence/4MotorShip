#!/usr/bin/env python
# coding=utf-8

import numpy as np
from numpy.lib.function_base import diff
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
        self.radius = 0.5
        
        self.factor12 = self.asv.asv_a / (np.power(self.asv.asv_a,2) + np.power(self.asv.asv_b,2))
        self.factor34 = self.asv.asv_b / (np.power(self.asv.asv_a,2) + np.power(self.asv.asv_b,2))

        self.draw_ed = []
        self.draw_aim_theta = []
        self.draw_aim_v = []

        plt.ion()

        self.observation_space = spaces.Box(low=0, high=50, shape=(13,))
        self.action_space = spaces.Box(low=-6, high=6, shape=(2,))
        self.action_bound = np.array([6, 3])
        self.torque_bound = np.array([12, 6])
        self.force_bound = 6
    
    def reset(self):
        """重设环境状态
            将目标点重置到(0, 0)之后，获取下一个目标点
            将船只重置为(0, 0)
        """
        self.aim.reset()
        begin_pos, begin_v= self.aim.observation()
        asv_pos, asv_v = self.asv.reset_state(begin_pos)
        self.draw_aim_theta.append(begin_pos[2])
        self.draw_aim_v.append(begin_v)
        self.draw_ed.append(math.sqrt(np.sum(np.power(asv_pos[0:2]-begin_pos[0:2],2))))
        self.aim.next_point()
    
        plt.ioff()
        plt.clf()
        plt.ion()

        self.draw_ed = []
        self.draw_aim_theta = []
        self.draw_aim_v = []
        return self.get_state()

    def get_state(self):
        """S: d, del_theta, l, del_u, del_v, del_r, a1, a2, a3, a4 

        d : diatance between asv and line which defined by last aim point and now aim point 

        del_theta: difference between target course angle course angle of asv(clockwise is positive) 

        l: distance between asv and now aim point
        """
        aim_pos, aim_v = self.aim.observation()
        asv_pos, asv_v = self.asv.observation()

        dx = self.getDx(asv_pos[0],asv_pos[1],aim_pos[0],aim_pos[1],aim_pos[2])
        dy = self.getDy(asv_pos[0],asv_pos[1],aim_pos[0],aim_pos[1],aim_pos[2])

        target_theta = self.targetCouseAngle(asv_pos[0],asv_pos[1],self.radius,aim_pos[0],aim_pos[1],aim_pos[2])
        asv_theta = asv_pos[2]
        del_theta = (0 - np.sign(target_theta - asv_theta)) * (math.pi * 2 - abs(target_theta - asv_theta)) if \
            abs(target_theta - asv_theta) > math.pi else target_theta - asv_theta
        
        # del_v = aim_v - asv_v

        # temp = np.array([dx, dy, asv_pos[2], aim_pos[2]])
        # # state = np.concatenate((temp, asv_v, del_v, self.asv.motor.data), axis=0)
        # state = np.concatenate((temp, asv_v, aim_v, self.asv.motor.data), axis=0)

        # del_theta_abs = math.atan2(aim_pos[1] - asv_pos[1], aim_pos[0] - asv_pos[0])

        a = np.array([dx, dy, del_theta, asv_pos[2], aim_pos[2]])
        state = np.concatenate((a, asv_v, aim_v, self.asv.torque), axis=0)
        return state

    def get_done(self):
        """If after asv take follow step the distance between asv and aim > r, announce episode DONE"""
        if self.l_after_a > 1:
            return True
        return False
        
    def get_reward(self):

        aim_pos, aim_v = self.aim.observation()
        asv_pos, asv_v = self.asv.observation()

        l = math.sqrt(np.sum(np.power((asv_pos[0:2] - aim_pos[0:2]), 2)))

        target_theta = self.targetCouseAngle(asv_pos[0],asv_pos[1],self.radius,aim_pos[0],aim_pos[1],aim_pos[2])
        asv_theta = asv_pos[2]
        del_theta = (0 - np.sign(target_theta - asv_theta)) * (math.pi * 2 - abs(target_theta - asv_theta)) if \
            abs(target_theta - asv_theta) > math.pi else target_theta - asv_theta
        # print(f'target_theta:{target_theta}, asv_theta:{asv_theta}, del_theta:{del_theta}')
        
        del_u,del_v,del_r = aim_v - asv_v

        del_l = self.l_before_a - l

        if del_l > 0:
            r_l = np.power(2, -10 * l) - 1
        else:
            r_l = -3

        del_theta = abs(del_theta)
        if del_theta > math.pi/2:
            r_theta = -np.exp(3*(del_theta-math.pi)) -1
        else:
            r_theta = np.exp(-3 * del_theta)

        # if del_theta > math.pi/2:
        #     r_theta = -6
        # else:
        #     r_theta = math.cos(del_theta)

        r1 = r_l + 0.5 * r_theta
        # print(f'del_theta:{del_theta}, r_theta:{r_theta}, r_l:{r_l}, l:{l}')

        error_v = 5 * np.power(del_u,2) + 30 * np.power(del_v,2) + 0.1 * np.power(del_r,2)
        r2 = np.exp(-3 * error_v) - 1

        sum_a = np.sum(np.power(self.asv.motor.data,2))
        r3 =  np.exp(-sum_a/50) - 1

        torque_his = np.array(self.asv.asv_his_torque)
        a_nearby = torque_his[-min(20, len(torque_his)):,:]
        r4 = 0
        for i in range(2):
            std = np.nan_to_num(np.std(a_nearby[:,i], ddof=1))
            r4 += 0.5 * (np.exp(-std) - 1)

        torque_nearby = torque_his[-min(5, len(torque_his)):,:]
        r5 = 0
        for i in range(len(torque_nearby[0])):
            torque_nearby_sep = torque_nearby[:,i]
            diff_abs = abs(torque_nearby_sep[-1] - torque_nearby_sep[0])
            diff_sum = 0
            if len(torque_nearby_sep) == 1:
                er_sep = 1
            else:
                for j in range(len(torque_nearby_sep)-1):
                    diff_sum += abs(torque_nearby_sep[j+1] - torque_nearby_sep[j])
                if diff_sum == 0:
                    er_sep = 1
                else:
                    er_sep = diff_abs / diff_sum
            temp = 0.2 * (np.exp(er_sep - 1) - 1)
            # print(f'torque:{torque_nearby_sep}, diff_abs:{diff_abs}, diff_sum:{diff_sum}, er_sep:{er_sep}, temp:{temp}')
            r5 += temp

        # print(f'r1:{r1}, r2:{r2}, r3:{r3}, r4:{r4}')

        r =r1 + r2 + r3 + r4 + r5

        return r

    def get_reward_punish(self):
        return -30
        
    def step(self, action):
        # 注意因为reset中已经让aim移动，因此aim永远是asv要追逐的点
        aim_pos, aim_v= self.aim.observation()
         # 计算asv本步移动前和aim之间的距离
        asv_pos, asv_v= self.asv.observation()
        self.l_before_a = math.sqrt(np.sum(np.power((asv_pos[0:2] - aim_pos[0:2]), 2)))
        # 在获得action之后，让asv根据action移动
        forward_torque = self.asv.torque[0] + action[0]
        rotate_torque = self.asv.torque[1] + action[1]

        self.asv.torque[0] = min(self.torque_bound[0], max(-self.torque_bound[0], forward_torque))
        self.asv.torque[1] = min(self.torque_bound[1], max(-self.torque_bound[1], rotate_torque))
        cur_motor = np.array([self.asv.torque[0]/2+self.factor12*self.asv.torque[1], self.asv.torque[0]/2- \
            self.factor12*self.asv.torque[1], self.factor34*self.asv.torque[1], -self.factor34*self.asv.torque[1]])
        # self.asv.motor = cur_motor
        self.asv.motor = np.clip(cur_motor, -self.force_bound, self.force_bound)
        
        # 让asv移动后，当前asv坐标更新为移动后的坐标
        cur_asv_pos, cur_asv_v = self.asv.move()

        # 计算asv移动后和aim之间的距离
        self.l_after_a = math.sqrt(np.sum(np.power((cur_asv_pos[0:2] - aim_pos[0:2]), 2)))
        
        # 奖励应该是对于当前aim，以及移动以后的asv计算
        done = self.get_done()
        # 注意奖励永远是根据当前aim坐标和当前asv坐标计算，当前aim尚未移动
        if done:
            reward = self.get_reward_punish()
        else:
            reward = self.get_reward()
        
        # draw数据准备
        self.draw_aim_theta.append(aim_pos[2])
        self.draw_aim_v.append(aim_v)
        self.draw_ed.append(math.sqrt(np.sum(np.power(cur_asv_pos[0:2]-aim_pos[0:2],2))))

        # 计算完奖励之后，可以移动aim坐标
        while True:
            cur_aim_pos, cur_aim_v = self.aim.next_point()
            dy = self.getDy(cur_asv_pos[0],cur_asv_pos[1],cur_aim_pos[0],cur_aim_pos[1],cur_aim_pos[2])
            if dy < 0:
                break
            else:
                reward = reward - 1  #如果aim点需多次移动，每多移一次reward-1
        # 此时aim已经是下一个要追逐的点，可以计算state
        state = self.get_state()

        return state, reward, done, ''

    def render(self):
        aim_his_pos = np.array(self.aim.aim_his_pos)
        draw_aim_theta = np.array(self.draw_aim_theta)
        draw_aim_v = np.array(self.draw_aim_v)
        draw_ed = np.array(self.draw_ed)
        asv_his_pos = np.array(self.asv.asv_his_pos)
        asv_his_v = np.array(self.asv.asv_his_v)
        motor_his = np.array(self.asv.asv_his_motor)
        torque_his = np.array(self.asv.asv_his_torque)

        plt.clf()

        # 绘制轨迹图
        plt.subplot(3,2,1)
        # 绘制aim
        plt.plot(*zip(*aim_his_pos[:,[0,1]]), 'y', label='aim')
        # 绘制asv
        # for i in range(len(asv_his_pos)):
        #     asv_his_pos[i][0] += np.random.normal(0, 0.02)
        #     asv_his_pos[i][1] += np.random.normal(0, 0.02)

        plt.plot(*zip(*asv_his_pos[:,[0,1]]), 'b', label='asv')
        # my_ticks = np.arange(-4, 8, 0.5)
        # plt.yticks(my_ticks)
        plt.title('x-y')
        plt.legend()

        # 绘制误差ed图
        plt.subplot(3,2,2)
        plt.plot(range(0, len(draw_ed)), draw_ed)
        plt.title('ed')

        # 绘制motor图
        plt.subplot(3,2,3)
        plt.plot(range(0, len(motor_his)), motor_his[:,0], label='f1')
        plt.plot(range(0, len(motor_his)), motor_his[:,1], label='f2')
        plt.plot(range(0, len(motor_his)), motor_his[:,2], label='f3')
        plt.plot(range(0, len(motor_his)), motor_his[:,3], label='f4')
        my_y_ticks = np.arange(-6, 7, 1)
        plt.yticks(my_y_ticks)
        plt.title('motor')
        plt.legend()

        # 绘制torque图
        plt.subplot(3,2,4)
        plt.plot(range(0, len(torque_his)), torque_his[:,0], label='forward')
        plt.plot(range(0, len(torque_his)), torque_his[:,1], label='rotate')
        plt.title('torque')
        plt.legend()

        # 绘制theta对比图
        plt.subplot(3,2,5)
        plt.plot(range(0, len(draw_aim_theta)), draw_aim_theta, label='aim')
        plt.plot(range(0, len(asv_his_pos)), asv_his_pos[:,2], label='asv')
        plt.title('theta')
        plt.legend()

        # 绘制asv的速度图
        plt.subplot(3,2,6)
        plt.plot(range(0, len(asv_his_v)), asv_his_v[:,0], label='u_asv')
        plt.plot(range(0, len(draw_aim_v)), draw_aim_v[:,0], label='u_aim')
        plt.plot(range(0, len(asv_his_v)), asv_his_v[:,1], label='v_asv')
        plt.plot(range(0, len(draw_aim_v)), draw_aim_v[:,1], label='v_aim')
        plt.plot(range(0, len(asv_his_v)), asv_his_v[:,2], label='r_asv')
        plt.plot(range(0, len(draw_aim_v)), draw_aim_v[:,2], label='r_aim')
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

    def pointToSegDist(self,pointX,pointY,lineX1,lineY1,lineX2,lineY2):
        """Distance between point and segment. 
        If the intersection of the perpendicular line from th point to the segment doesn't exist,
        return the distance from the point to the nearst endpoint of segment.

        :param pointX,pointY : the point
        :param lineX1,lineY1 : one endpoint of segment
        :param lineX2,lineY2 : the other endpoint of segment
        """
        cross = (lineX2 - lineX1) * (pointX - lineX1) + (lineY2 - lineY1) * (pointY - lineY1) #|AB*AP|：矢量乘
        if (cross <= 0):
            return math.sqrt((pointX - lineX1) * (pointX - lineX1) + (pointY - lineY1) * (pointY - lineY1)) #是|AP|：矢量的大小
        
        d2 = (lineX2 - lineX1) * (lineX2 - lineX1) + (lineY2 - lineY1) * (lineY2 - lineY1) #|AB|^2：矢量AB的大小的平方
        if (cross >= d2):
            return math.sqrt((pointX - lineX2) * (pointX - lineX2) + (pointY - lineY2) * (pointY - lineY2)) #是|BP|：矢量的大小
        
        r = cross / d2 #相似三角形原理求出c点的坐标
        px = lineX1 + (lineX2 - lineX1) * r
        py = lineY1 + (lineY2 - lineY1) * r
        return math.sqrt((pointX - px) * (pointX - px) + (py - pointY) * (py - pointY))
    
    def pointToLineDist(self,pointX,pointY,lineX1,lineY1,lineX2,lineY2):
        """Distance between point and line. The line is defined by two mark point on it.
        """
        a=lineY2-lineY1
        b=lineX1-lineX2
        c=lineX2*lineY1-lineX1*lineY2
        dis=(math.fabs(a*pointX+b*pointY+c))/(math.pow(a*a+b*b,0.5))
        return dis
    
    def pointLineSide(self,pointX,pointY,lineX1,lineY1,lineX2,lineY2):
        """Judge the point is on which side of the directional line.

        If the point can rotate clockwise to the line, return positive number.
        If the point can rotate counterclockwise to the line, return negative number.
        If the point is on the line, return zero.

        :param pointX,pointY : the point
        :param lineX1,lineY1 : starting point of the line vector
        :param lineX2,lineY2 : ending point of the line vector
        """
        s = (lineX1-pointX)*(lineY2-pointY)-(lineY1-pointY)*(lineX2-pointX)
        return s

    def footOfPOintPerpendicularToLine(self,pointX,pointY,lineX1,lineY1,theta):
        k = math.tan(theta)

        if abs(abs(theta) - math.pi/2) < 1e-6:  # 直线垂直于x轴
            return lineX1,pointY
        if abs(theta - math.pi) < 1e-6 or abs(theta - 0.0) < 1e-6:   #直线垂直于y轴
            return pointX,lineY1
        if abs(pointY - (k*(pointX-lineX1) + lineY1)) < 1e-6:   #点在直线上
            return pointX,pointY
        
        x = (np.power(k,2) * lineX1 + k * (pointY - lineY1) + pointX) / (np.power(k,2) + 1)
        y = k * (x - lineX1) + lineY1
        return x,y

    def getDx(self,pointX,pointY,lineX1,lineY1,theta):
        foot_point_x, foot_point_y = self.footOfPOintPerpendicularToLine(pointX,pointY,lineX1,lineY1,theta)
        dx = np.sqrt(np.power(foot_point_x - pointX,2) + np.power(foot_point_y - pointY,2))
        signal_dx = (foot_point_x-lineX1) * (pointY-lineY1) - (foot_point_y-lineY1) * (pointX-lineX1)
        if signal_dx > 0:
            dx = dx
        elif signal_dx < 0:
            dx = -dx
        else:
            dx = 0
        return dx

    def getDy(self,pointX,pointY,lineX1,lineY1,theta):
        foot_point_x, foot_point_y = self.footOfPOintPerpendicularToLine(pointX,pointY,lineX1,lineY1,theta)
        dy = np.sqrt(np.power(foot_point_x - lineX1,2) + np.power(foot_point_y - lineY1,2))
        theta_foot_aim = math.atan2(foot_point_y - lineY1,foot_point_x - lineX1)
        if abs(theta_foot_aim - theta) < 1e-6:   
            dy = dy
        else:
            dy = -dy
        return dy

    def targetCouseAngle(self,pointX,pointY,r,lineX1,lineY1,theta):
        d = self.getDx(pointX,pointY,lineX1,lineY1,theta)
        if abs(d)/r >1 : 
            adjust_angle = math.asin(1)
        else:
            adjust_angle = math.asin(abs(d) / r)

        if abs(d) < 1e-6:
            target_angle = theta
        elif d < 0:
            target_angle = theta - adjust_angle
        else:
            target_angle = theta + adjust_angle
        return target_angle
     