
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

phy_data_sheet = pd.read_excel('D:\ASV project file\save_data1125\\20201123 135900 sin.xlsx', sheet_name=[0,1,2,3])
phy_aim_state = phy_data_sheet[0].values[:,1:]
phy_asv_state = phy_data_sheet[1].values[:,1:] #-1
phy_motor = phy_data_sheet[2].values[:,1:]
phy_torque = phy_data_sheet[3].values[:,1:]

phy_aim_his_pos = np.array(phy_aim_state[:,:2])
phy_aim_his_theta = np.array(phy_aim_state[:,2])
phy_aim_his_v = np.array(phy_aim_state[:,3:])
phy_asv_his_pos = np.array(phy_asv_state[:,:2])
phy_asv_his_v = np.array(phy_asv_state[:,3:])
phy_asv_his_theta = np.array(phy_asv_state[:,2])
phy_motor_his = np.array(phy_motor)
phy_torque_his = np.array(phy_torque)
phy_ed = []
phy_step = len(phy_aim_his_pos)

for i in range(len(phy_asv_his_pos)-1):
    x_error = pow(phy_aim_his_pos[i][0] - phy_asv_his_pos[i+1][0], 2)
    y_error = pow(phy_aim_his_pos[i][1] - phy_asv_his_pos[i+1][1], 2)
    error = np.sqrt(x_error + y_error)
    phy_ed.append(error)

sim_data_sheet = pd.read_excel('D:\\Master Study Material\\ASVProject\\4MotorShip\\save_data\\20201202 224503.xlsx', sheet_name=[0,1,2,3])
sim_aim_state = sim_data_sheet[0].values[:phy_step,1:]
sim_asv_state = sim_data_sheet[1].values[:phy_step+1,1:] #-1
sim_motor = sim_data_sheet[2].values[:phy_step-1,1:]
sim_torque = sim_data_sheet[3].values[:phy_step-1,1:]

sim_aim_his_pos = np.array(sim_aim_state[:,:2])
sim_aim_his_theta = np.array(sim_aim_state[:,2])
sim_aim_his_v = np.array(sim_aim_state[:,3:])
sim_asv_his_pos = np.array(sim_asv_state[:,:2])
sim_asv_his_v = np.array(sim_asv_state[:,3:])
sim_asv_his_theta = np.array(sim_asv_state[:,2])
sim_motor_his = np.array(sim_motor)
sim_torque_his = np.array(sim_torque)
sim_ed = []

for i in range(len(sim_asv_his_pos)-1):
    x_error = pow(sim_aim_his_pos[i][0] - sim_asv_his_pos[i+1][0], 2)
    y_error = pow(sim_aim_his_pos[i][1] - sim_asv_his_pos[i+1][1], 2)
    error = np.sqrt(x_error + y_error)
    sim_ed.append(error)

fig = plt.figure()
plt.ion()

for i in range(len(phy_asv_his_pos)):
    plt.clf()
    # 绘制轨迹图
    plt.subplot(3,2,1)
    # 绘制asv
    plt.plot(*zip(*phy_asv_his_pos[:i+1,[0,1]]), 'b', label='phy', linewidth=2)
    plt.plot(*zip(*sim_asv_his_pos[:i+1,[0,1]]), 'r', label='sim', linewidth=2, linestyle='-.')
    # 绘制aim
    plt.plot(*zip(*phy_aim_his_pos[:i+1,[0,1]]), 'y', label='aim', linewidth=2, linestyle='--')
    plt.title('X-Y')
    plt.xlabel('X[m]')
    plt.ylabel('Y[m]')
    plt.grid()#添加网格
    plt.legend(loc='lower right', fontsize=10, borderpad=0.3, labelspacing=0.3)

    # 绘制误差ed图
    plt.subplot(3,2,2)
    plt.plot(np.arange(0, len(phy_ed[:i+1]) / 10, 0.1), phy_ed[:i+1], label='phy', linewidth=2)
    plt.plot(np.arange(0, len(sim_ed[:i+1]) / 10, 0.1), sim_ed[:i+1], label='sim', linewidth=2, linestyle='-.')
    plt.title('Distance Error')
    plt.xlabel('Time[s]')
    plt.ylabel('Distance Error[m]')
    plt.grid()#添加网格
    plt.legend(loc='lower right', fontsize=10, borderpad=0.3, labelspacing=0.3)

    # 绘制torque图
    plt.subplot(3,2,4)
    plt.plot(np.arange(0,len(phy_torque_his[:i+1,:]) / 10, 0.1), phy_torque_his[:i+1,0], label='phy_forward', linewidth=2)
    plt.plot(np.arange(0,len(phy_torque_his[:i+1,:]) / 10, 0.1), phy_torque_his[:i+1,1], label='phy_rotate', linewidth=2)
    plt.plot(np.arange(0,len(sim_torque_his[:i+1,:]) / 10, 0.1), sim_torque_his[:i+1,0], label='sim_forward', linewidth=2, linestyle='-.')
    plt.plot(np.arange(0,len(sim_torque_his[:i+1,:]) / 10, 0.1), sim_torque_his[:i+1,1], label='sim_rotate', linewidth=2, linestyle='-.')
    plt.title('Torque')
    plt.xlabel('Time[s]')
    plt.ylabel('Torque')
    plt.grid()#添加网格
    plt.legend(loc='lower right', fontsize=10, borderpad=0.3, labelspacing=0.3)

    # 绘制theta对比图
    plt.subplot(3,2,5)
    plt.plot(np.arange(0,len(phy_aim_his_theta[:i+1]) / 10, 0.1), phy_aim_his_theta[:i+1], label='aim', linewidth=2, linestyle='--')
    plt.plot(np.arange(0,len(phy_aim_his_theta[:i+1]) / 10, 0.1), phy_asv_his_theta[1:i+2], label='phy', linewidth=2)
    plt.plot(np.arange(0,len(sim_aim_his_theta[:i+1]) / 10, 0.1), sim_asv_his_theta[1:i+2], label='sim', linewidth=2, linestyle='-.')
    plt.title('Heading Angle')
    plt.xlabel('Time[s]')
    plt.ylabel('Heading Angle[rad]')
    plt.grid()#添加网格
    plt.legend(loc='lower right', fontsize=10, borderpad=0.3, labelspacing=0.3)

    # 绘制asv的速度图
    plt.subplot(3,2,6)
    plt.plot(np.arange(0,len(phy_asv_his_v[:i+1,:]) / 10, 0.1), phy_asv_his_v[:i+1,0], label='u_phy', linewidth=2)
    plt.plot(np.arange(0,len(sim_asv_his_v[:i+1,:]) / 10, 0.1), sim_asv_his_v[:i+1,0], label='u_sim', linewidth=2, linestyle='-.')
    plt.plot(np.arange(0,len(phy_aim_his_v[:i+1,:]) / 10, 0.1), phy_aim_his_v[:i+1,0], label='u_target', linewidth=2, linestyle='--')
    plt.plot(np.arange(0,len(phy_asv_his_v[:i+1,:]) / 10, 0.1), phy_asv_his_v[:i+1,1], label='v_phy', linewidth=2)
    plt.plot(np.arange(0,len(sim_asv_his_v[:i+1,:]) / 10, 0.1), sim_asv_his_v[:i+1,1], label='v_sim', linewidth=2, linestyle='-.')
    plt.plot(np.arange(0,len(phy_aim_his_v[:i+1,:]) / 10, 0.1), phy_aim_his_v[:i+1,1], label='v_target', linewidth=2, linestyle='--')
    plt.plot(np.arange(0,len(phy_asv_his_v[:i+1,:]) / 10, 0.1), phy_asv_his_v[:i+1,2], label='r_phy', linewidth=2)
    plt.plot(np.arange(0,len(sim_asv_his_v[:i+1,:]) / 10, 0.1), sim_asv_his_v[:i+1,2], label='r_sim', linewidth=2, linestyle='-.')
    plt.plot(np.arange(0,len(phy_aim_his_v[:i+1,:]) / 10, 0.1), phy_aim_his_v[:i+1,2], label='r_target', linewidth=2, linestyle='--')
    plt.title('Velocity')
    plt.xlabel('Time[s]')
    plt.ylabel('Velocity[m/s]')
    plt.grid()#添加网格
    plt.legend(loc='lower right', fontsize=8, borderpad=0.3, labelspacing=0.3)

    plt.pause(0.5)

plt.ioff()