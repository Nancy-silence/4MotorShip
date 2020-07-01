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
noise = NormalActionNoise(0*np.ones(1), 0.02*np.ones(1))   
# noise = OUActionNoise(0*np.ones(1), 0.1*np.ones(1))
x = np.arange(1000)
y = []
for i in range(1000):
    y.append(noise())

plt.plot(x,y)
plt.show()
    
