
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data_sheet = pd.read_excel('D:\ASV project file\save_data1125\\20201123 135900 sin.xlsx', sheet_name=[0,1,2,3])
aim_state = data_sheet[0].values[:,1:]
asv_state = data_sheet[1].values[:,1:] #-1
motor = data_sheet[2].values[:,1:]
torque = data_sheet[3].values[:,1:]

aim_his_pos = np.array(aim_state[:,:2])
aim_his_theta = np.array(aim_state[:,2])
aim_his_v = np.array(aim_state[:,3:])
asv_his_pos = np.array(asv_state[:,:2])
asv_his_v = np.array(asv_state[:,3:])
asv_his_theta = np.array(asv_state[:,2])
motor_his = np.array(motor)
torque_his = np.array(torque)
ed = []

for i in range(len(asv_his_pos)-1):
    x_error = pow(aim_his_pos[i][0] - asv_his_pos[i+1][0], 2)
    y_error = pow(aim_his_pos[i][1] - asv_his_pos[i+1][1], 2)
    error = np.sqrt(x_error + y_error)
    ed.append(error)

fig = plt.figure()
plt.ion()

for i in range(len(asv_his_pos)):
    plt.clf()
    # 绘制轨迹图
    plt.subplot(3,2,1)
    # 绘制asv
    plt.plot(*zip(*asv_his_pos[:i+1,[0,1]]), 'b', label='USV', linewidth=2)
    # 绘制aim
    plt.plot(*zip(*aim_his_pos[:i+1,[0,1]]), 'y', label='aim', linewidth=2, linestyle='--')
    plt.title('X-Y')
    plt.xlabel('X[m]')
    plt.ylabel('Y[m]')
    plt.grid()#添加网格
    plt.legend()

    # 绘制误差ed图
    plt.subplot(3,2,2)
    plt.plot(np.arange(0, len(ed[:i+1]) / 10, 0.1), ed[:i+1], linewidth=2)
    plt.title('Distance Error')
    plt.xlabel('Time[s]')
    plt.ylabel('Distance Error[m]')
    plt.grid()#添加网格

    # 绘制torque图
    plt.subplot(3,2,4)
    plt.plot(np.arange(0,len(torque_his[:i+1,:]) / 10, 0.1), torque_his[:i+1,0], label='forward', linewidth=2)
    # plt.plot(range(len(torque_his) - len(smooth_torque), len(torque_his)), smooth_torque, label='forward smooth')
    plt.plot(np.arange(0,len(torque_his[:i+1,:]) / 10, 0.1), torque_his[:i+1,1], label='rotate', linewidth=2)
    plt.title('Torque')
    plt.xlabel('Time[s]')
    plt.ylabel('Torque')
    plt.grid()#添加网格
    plt.legend()

    # 绘制theta对比图
    plt.subplot(3,2,5)
    plt.plot(np.arange(0,len(aim_his_theta[:i+1]) / 10, 0.1), aim_his_theta[:i+1], label='aim', linewidth=2, linestyle='--')
    plt.plot(np.arange(0,len(aim_his_theta[:i+1]) / 10, 0.1), asv_his_theta[1:i+2], label='USV', linewidth=2)
    plt.title('Heading Angle')
    plt.xlabel('Time[s]')
    plt.ylabel('Heading Angle[rad]')
    plt.grid()#添加网格
    plt.legend()

    # 绘制asv的速度图
    plt.subplot(3,2,6)
    plt.plot(np.arange(0,len(asv_his_v[:i+1,:]) / 10, 0.1), asv_his_v[:i+1,0], label='u_USV', linewidth=2)
    plt.plot(np.arange(0,len(aim_his_v[:i+1,:]) / 10, 0.1), aim_his_v[:i+1,0], label='u_target', linewidth=2, linestyle='--')
    plt.plot(np.arange(0,len(asv_his_v[:i+1,:]) / 10, 0.1), asv_his_v[:i+1,1], label='v_USV', linewidth=2)
    plt.plot(np.arange(0,len(aim_his_v[:i+1,:]) / 10, 0.1), aim_his_v[:i+1,1], label='v_target', linewidth=2, linestyle='--')
    plt.plot(np.arange(0,len(asv_his_v[:i+1,:]) / 10, 0.1), asv_his_v[:i+1,2], label='r_USV', linewidth=2)
    plt.plot(np.arange(0,len(aim_his_v[:i+1,:]) / 10, 0.1), aim_his_v[:i+1,2], label='r_target', linewidth=2, linestyle='--')
    plt.title('Velocity')
    plt.xlabel('Time[s]')
    plt.ylabel('Velocity[m/s]')
    plt.grid()#添加网格
    plt.legend(loc='lower right', fontsize=10, borderpad=0.3, labelspacing=0.3)

    plt.pause(0.5)

plt.ioff()