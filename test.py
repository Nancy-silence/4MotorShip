# import numpy as np
# import math
# import matplotlib.pyplot as plt

# x = []
# x_v = []
# x_a = []
# y = []
# y_v = []
# y_a = []
# u_list = []
# theta_list = []
# t = np.arange(0,30.1,0.1)
# for i in t :
#     # sx = 6*(np.power(i,3)*10/np.power(30,3) - np.power(i,4)*15/np.power(30,4) + np.power(i,5)*6/np.power(30,5))
#     sx = 6*i*i/900
#     x.append(sx)
#     sy = -3*math.cos(sx)+3
    
#     # sx_v = 6*(np.power(i,2)*30/np.power(30,3) - np.power(i,3)*60/np.power(30,4) + np.power(i,4)*30/np.power(30,5))
#     sx_v = 6*i*2/900
#     u = np.power(sx_v,2)
#     x_v.append(sx_v)
#     sy_v = 3*math.sin(sx)*sx_v
#     u += np.power(sy_v,2)
#     u = math.sqrt(u)

#     theta = math.atan2(sy_v, sx_v)
    
#     # sx_a = 6*(i*60/np.power(30,3) - np.power(i,2)*180/np.power(30,4) + np.power(i,3)*120/np.power(30,5))
#     sx_a = 6*2/900
#     x_a.append(sx_a)
#     sy_a = 3*(math.cos(sx)*sx_v*sx_v + math.sin(sx)*sx_a)
#     y.append(sy)
#     y_v.append(sy_v)
#     y_a.append(sy_a)
#     u_list.append(u)
#     theta_list.append(theta)

# plt.subplot(2,4,1)
# plt.plot(x,y)
# plt.subplot(2,4,2)
# plt.plot(t,x,label="x")
# plt.plot(t,y,label="y")
# plt.legend()
# plt.subplot(2,4,3)
# plt.plot(t,x_v,label="x")
# plt.plot(t,y_v,label="y")
# plt.plot(t,u_list,label="u")
# plt.legend()
# plt.subplot(2,4,4)
# plt.plot(t,x_a,label="x")
# plt.plot(t,y_a,label="y")
# plt.legend()

# plt.subplot(2,4,5)
# plt.plot(t,theta_list)
# plt.show()

# 测试OU噪声和高斯噪声的区别及各自表现

from Utils import OUActionNoise,NormalActionNoise
import matplotlib.pyplot as plt
import numpy as np
# noise = NormalActionNoise(0*np.ones(1), 0.3*np.ones(1))   
noise = OUActionNoise(0*np.ones(1), 0.01 * np.ones(1))
x = np.arange(1000)
y = []
for i in range(1000):
    y.append(noise())

plt.plot(x,y)
plt.show()

# import numpy as np
# # error = np.array([1,2,3,4,5,6])
# # print(error[-1])
# import math
# def pointToSegDist(pointX,pointY,lineX1,lineY1,lineX2,lineY2):
#     """Distance between point and segment. 
#     If the intersection of the perpendicular line from th point to the segment doesn't exist,
#     return the distance from the point to the nearst endpoint of segment.

#     :param pointX,pointY : the point
#     :param lineX1,lineY1 : one endpoint of segment
#     :param lineX2,lineY2 : the other endpoint of segment
#     """
#     cross = (lineX2 - lineX1) * (pointX - lineX1) + (lineY2 - lineY1) * (pointY - lineY1) #|AB*AP|：矢量乘
#     if (cross <= 0):
#         return math.sqrt((pointX - lineX1) * (pointX - lineX1) + (pointY - lineY1) * (pointY - lineY1)) #是|AP|：矢量的大小
    
#     d2 = (lineX2 - lineX1) * (lineX2 - lineX1) + (lineY2 - lineY1) * (lineY2 - lineY1) #|AB|^2：矢量AB的大小的平方
#     if (cross >= d2):
#         return math.sqrt((pointX - lineX2) * (pointX - lineX2) + (pointY - lineY2) * (pointY - lineY2)) #是|BP|：矢量的大小
    
#     r = cross / d2 #相似三角形原理求出c点的坐标
#     px = lineX1 + (lineX2 - lineX1) * r
#     py = lineY1 + (lineY2 - lineY1) * r
#     return math.sqrt((pointX - px) * (pointX - px) + (py - pointY) * (py - pointY))

# def pointToLineDist(pointX,pointY,lineX1,lineY1,lineX2,lineY2):
#     """Distance between point and line. The line is defined by two mark point on it.
#     """
#     a=lineY2-lineY1
#     b=lineX1-lineX2
#     c=lineX2*lineY1-lineX1*lineY2
#     dis=(math.fabs(a*pointX+b*pointY+c))/(math.pow(a*a+b*b,0.5))
#     return dis

# def pointLineSide(pointX,pointY,lineX1,lineY1,lineX2,lineY2):
#     """Judge the point is on which side of the directional line.

#     If the point can rotate clockwise to the line, return positive number.
#     If the point can rotate counterclockwise to the line, return negative number.
#     If the point is on the line, return zero.

#     :param pointX,pointY : the point
#     :param lineX1,lineY1 : starting point of the line vector
#     :param lineX2,lineY2 : ending point of the line vector
#     """
#     s = (lineX1-pointX)*(lineY2-pointY)-(lineY1-pointY)*(lineX2-pointX)
#     return s

# def targetCouseAngle(pointX,pointY,r,lineX1,lineY1,lineX2,lineY2):
#     fai_path = math.atan2((lineY2-lineY1), (lineX2-lineX1))
#     print(fai_path)
#     side = pointLineSide(pointX,pointY,lineX1,lineY1,lineX2,lineY2)
#     d = pointToLineDist(pointX,pointY,lineX1,lineY1,lineX2,lineY2)
#     adjust_angle = math.asin(d / r)
#     print(adjust_angle)
#     if side > 0:
#         target_angle = fai_path - adjust_angle
#     elif side < 0:
#         target_angle = fai_path + adjust_angle
#     else:
#         target_angle = fai_path
#     return target_angle

# def footOfPerpendicular(pointX,pointY,lineX1,lineY1,theta):
#     k = math.tan(theta)

#     if abs(abs(theta) - math.pi/2) < 1e-6:  # 直线垂直于x轴
#         return lineX1,pointY
#     if abs(theta - math.pi) < 1e-6 or abs(theta - 0.0) < 1e-6:   #直线垂直于y轴
#         return pointX,lineY1
#     if abs((lineY1-pointY)/(lineX1-pointX) - k) < 1e-6:
#         return pointX,pointY
    
#     x = (np.power(k,2) * lineX1 + k * (pointY - lineY1) + pointX) / (np.power(k,2) + 1)
#     y = k * (x - lineX1) + lineY1
#     return x,y

# def getDx(pointX,pointY,lineX1,lineY1,theta):
#     k = math.tan(theta)

#     if abs(abs(theta) - math.pi/2) < 1e-6:  # 直线垂直于x轴
#         x,y = lineX1,pointY
#     if abs(theta - math.pi) < 1e-6 or abs(theta - 0.0) < 1e-6:   #直线垂直于y轴
#         x,y = pointX,lineY1
#     if abs(pointY - (k*(pointX-lineX1) + lineY1)) < 1e-6:
#         x,y = pointX,pointY
#     else:
#         x = (np.power(k,2) * lineX1 + k * (pointY - lineY1) + pointX) / (np.power(k,2) + 1)
#         y = k * (x - lineX1) + lineY1

#     foot_point_x, foot_point_y = x,y
#     dx = np.sqrt(np.power(foot_point_x - pointX,2) + np.power(foot_point_y - pointY,2))
#     signal_dx = (foot_point_x-lineX1) * (pointY-lineY1) - (foot_point_y-lineY1) * (pointX-lineX1)
#     if signal_dx > 0:
#         dx = dx
#     elif signal_dx < 0:
#         dx = -dx
#     else:
#         dx = 0
#     return dx

# def getDy(pointX,pointY,lineX1,lineY1,theta):
#     k = math.tan(theta)

#     if abs(abs(theta) - math.pi/2) < 1e-6:  # 直线垂直于x轴
#         x,y = lineX1,pointY
#     if abs(theta - math.pi) < 1e-6 or abs(theta - 0.0) < 1e-6:   #直线垂直于y轴
#         x,y = pointX,lineY1
#     if abs(pointY - (k*(pointX-lineX1) + lineY1)) < 1e-6:
#         x,y = pointX,pointY
#     else:
#         x = (np.power(k,2) * lineX1 + k * (pointY - lineY1) + pointX) / (np.power(k,2) + 1)
#         y = k * (x - lineX1) + lineY1
        
#     foot_point_x, foot_point_y = x,y

#     dy = np.sqrt(np.power(foot_point_x - lineX1,2) + np.power(foot_point_y - lineY1,2))
#     theta_foot_aim = math.atan2(foot_point_y - lineY1,foot_point_x - lineX1)
#     if abs(theta_foot_aim - theta) < 1e-6:   
#         dy = dy
#     else:
#         dy = -dy
#     return dy

# if __name__ == '__main__':
#     print(getDx(-1,-1,1,-1,-math.pi/4))
